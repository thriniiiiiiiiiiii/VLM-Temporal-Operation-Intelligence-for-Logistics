# Failure Modes & Recovery Procedures

## 1. CUDA OOM During Training
**Symptoms:** `RuntimeError: CUDA out of memory`
**Recovery:**
1. Reduce `batch_size` from 2 to 1
2. Reduce `frames_per_clip` from 8 to 4
3. Reduce LoRA rank from 16 to 8
4. Verify `gradient_checkpointing=True` is enabled

## 2. Kaggle/Colab Session Crash
**Symptoms:** Session disconnects mid-training
**Recovery:**
1. Reconnect to runtime
2. Re-mount Google Drive / re-attach Kaggle persistent storage
3. Re-run setup cells
4. Training auto-resumes from latest `checkpoint-*` via `resume_from_checkpoint=True`

## 3. NaN Loss
**Symptoms:** Loss becomes `nan` or `inf`
**Recovery:**
1. Reduce learning rate by 10x (0.0002 → 0.00002)
2. Ensure `max_grad_norm=1.0` is set
3. Check for corrupted training samples (malformed images or empty JSON)
4. Resume from the last stable checkpoint

## 4. Tape ↔ Pack Confusion (Primary ML Failure)
**Symptoms:** 18-31% of Tape clips misclassified as Pack
**Mitigation:**
- Expand boundary window from ±0.5s to ±1.0s for Tape/Pack boundaries
- Add workflow grammar constraint at decoding time
- Augment Tape training clips with brightness/contrast variations

## 5. API Model Loading Failure
**Symptoms:** `/ready` returns 503
**Recovery:**
1. Check `TRANSFORMERS_CACHE` and `HF_HOME` env vars point to valid cache
2. Ensure adapter checkpoint path is correct (`ADAPTER_PATH`)
3. Verify GPU is available: `torch.cuda.is_available()`

## 6. Malformed VLM Output
**Symptoms:** Model returns unstructured text instead of JSON
**Recovery:** The API uses a 4-stage fallback parser:
1. Direct `json.loads()` on raw output
2. Extract from markdown code blocks
3. Regex extraction of first `{...}` block
4. Heuristic keyword matching (last resort, confidence=0.0)
