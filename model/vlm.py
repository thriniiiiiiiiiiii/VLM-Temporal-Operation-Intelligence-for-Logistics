"""
model/vlm.py
────────────
Qwen2.5-VL model loading, quantization config, and inference engine.

Design principles:
  - Load once, serve many: model stays in memory across requests
  - 4-bit quantization: reduces 2B model from ~4 GB to ~2 GB VRAM
  - Robust JSON parsing: handles model hallucinations and partial outputs
  - Stateless inference: each request is independent
"""

import io
import json
import re
import time
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from loguru import logger
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
try:
    from qwen_vl_utils import process_vision_info
    _HAS_QWEN_UTILS = True
except ImportError:
    _HAS_QWEN_UTILS = False
    logger.warning("qwen-vl-utils not installed — using manual vision processing")

from data_pipeline import OPERATION_CLASSES, WORKFLOW_TRANSITIONS, SYSTEM_PROMPT


# ── Quantization Config ───────────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ── Model Singleton ───────────────────────────────────────────────────────────

class VLMEngine:
    """
    Thread-safe VLM inference engine.
    Loads model once and handles all inference requests.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-2B-Instruct",
        adapter_path: Optional[str] = None,
        quantize: bool = True,
        device_map: str = "auto",
        max_new_tokens: int = 256,
    ):
        self.model_path     = model_path
        self.adapter_path   = adapter_path
        self.quantize       = quantize
        self.device_map     = device_map
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        logger.info(f"Loading VLM from: {self.model_path}")
        t0 = time.time()

        kwargs = {
            "trust_remote_code": True,
            "device_map": self.device_map,
        }
        if self.quantize:
            kwargs["quantization_config"] = get_bnb_config()
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, **kwargs
        )

        # Load LoRA adapter if fine-tuned model
        if self.adapter_path and Path(self.adapter_path).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, self.adapter_path
            )
            logger.info(f"LoRA adapter loaded from: {self.adapter_path}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model.eval()
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")
        self._log_memory()

    def unload(self):
        import torch
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VLM model unloaded and CUDA cache cleared.")
        self._log_memory()

    def _log_memory(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved  = torch.cuda.memory_reserved() / 1e9
            logger.info(
                f"GPU memory — Allocated: {allocated:.2f} GB | "
                f"Reserved: {reserved:.2f} GB"
            )

    def predict(
        self,
        frames: list[Image.Image],
        clip_id: str = "unknown",
    ) -> dict:
        """
        Run inference on a list of PIL frames.
        Returns parsed prediction dict or fallback on parse failure.
        """
        t0 = time.time()

        # Build conversation
        content = []
        for img in frames:
            content.append({"type": "image", "image": img})
        content.append({
            "type": "text",
            "text": (
                f"These {len(frames)} frames are from a 5-second warehouse "
                "packaging video clip (temporally ordered). Analyze the sequence "
                "and respond ONLY with a valid JSON object."
            )
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]

        # Tokenize
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if _HAS_QWEN_UTILS:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text_input],
                images=frames,
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.1,
            )

        # Decode only the newly generated tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_text  = self.processor.decode(generated[0], skip_special_tokens=True)

        latency_ms = int((time.time() - t0) * 1000)
        logger.debug(f"[{clip_id}] Inference latency: {latency_ms}ms")

        return self._parse_output(raw_text, clip_id, latency_ms)

    def _parse_output(self, raw: str, clip_id: str, latency_ms: int) -> dict:
        """
        Robustly parse model output into structured prediction.
        Handles: clean JSON, JSON wrapped in markdown, partial JSON,
                 and complete parsing failures.
        """
        # 1. Try direct parse
        pred = _try_json_parse(raw)

        # 2. Extract JSON from markdown code block
        if pred is None:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                pred = _try_json_parse(match.group(1))

        # 3. Extract first { ... } block
        if pred is None:
            match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if match:
                pred = _try_json_parse(match.group(0))

        # 4. Fallback with heuristic extraction
        if pred is None:
            logger.warning(f"[{clip_id}] JSON parse failed — applying heuristics")
            pred = _heuristic_extract(raw)

        # 5. Validate and sanitize
        pred = _sanitize_prediction(pred, clip_id)
        pred["clip_id"]     = clip_id
        pred["latency_ms"]  = latency_ms
        pred["raw_output"]  = raw[:200]  # store truncated raw for debugging

        return pred


def _try_json_parse(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _heuristic_extract(text: str) -> dict:
    """Last-resort extraction using regex patterns."""
    result = {}

    # Extract operation class
    for op in OPERATION_CLASSES:
        if op.lower() in text.lower():
            result["dominant_operation"] = op
            break

    # Extract frame numbers
    numbers = re.findall(r"\d+", text)
    ints    = [int(n) for n in numbers if 1 <= int(n) <= 200]
    if len(ints) >= 2:
        result["temporal_segment"] = {
            "start_frame": min(ints[:2]),
            "end_frame":   max(ints[:2]),
        }

    # Extract next operation
    for op in OPERATION_CLASSES:
        if f"next.*{op.lower()}" in text.lower() or f"{op.lower()}.*next" in text.lower():
            result["anticipated_next_operation"] = op
            break

    return result


def get_mock_prediction(clip_id: str) -> dict:
    """Return a realistic mock prediction for local testing."""
    import random
    from data_pipeline import OPERATION_CLASSES
    op = random.choice(OPERATION_CLASSES)
    next_op = random.choice(OPERATION_CLASSES)
    return {
        "dominant_operation": op,
        "temporal_segment": {"start_frame": random.randint(1, 10), "end_frame": random.randint(110, 150)},
        "anticipated_next_operation": next_op,
        "confidence": round(random.uniform(0.65, 0.98), 2),
    }

def _get_mock_prediction(clip_id: str) -> dict:
    return get_mock_prediction(clip_id)


def _sanitize_prediction(pred: dict, clip_id: str) -> dict:
    """Ensure all required fields exist and are valid."""
    # dominant_operation
    op = pred.get("dominant_operation", "Unknown")
    if op not in OPERATION_CLASSES:
        # Find closest match
        op_lower = op.lower()
        op = next(
            (c for c in OPERATION_CLASSES if c.lower() in op_lower),
            "Unknown"
        )
    pred["dominant_operation"] = op

    # temporal_segment
    seg = pred.get("temporal_segment", {})
    if not isinstance(seg, dict):
        seg = {}
    sf = _clamp(int(seg.get("start_frame", 1)),  1, 125)
    ef = _clamp(int(seg.get("end_frame",   125)), 1, 125)
    if sf >= ef:
        ef = min(125, sf + 10)
    pred["temporal_segment"] = {"start_frame": sf, "end_frame": ef}

    # anticipated_next_operation
    next_op = pred.get("anticipated_next_operation", "Unknown")
    if next_op not in OPERATION_CLASSES:
        # Use workflow grammar as prior
        valid_nexts = WORKFLOW_TRANSITIONS.get(op, ["Unknown"])
        next_op = valid_nexts[0] if valid_nexts else "Unknown"
    pred["anticipated_next_operation"] = next_op

    # confidence
    conf = pred.get("confidence", 0.5)
    pred["confidence"] = float(_clamp(conf, 0.0, 1.0))

    return pred


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))
