import requests
import time
import os
import subprocess
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"
TEST_VIDEO = "test_dummy.mp4"

def create_dummy_video(path: str):
    """Generate a 1-second white dummy MP4 using ffmpeg."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=white:s=336x336:d=1",
            "-pix_fmt", "yuv420p", path
        ], check=True, capture_output=True)
        print(f"✅ Created dummy video: {path}")
    except Exception as e:
        print(f"⚠️ Failed to create dummy video with ffmpeg: {e}")
        # Create a 0-byte file as fallback for schema testing
        Path(path).touch()

def run_test():
    print("🚀 Starting API Integration Test...")
    
    # 1. Health Check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        print("✅ /health check: OK")
    except Exception as e:
        print(f"❌ /health check FAILED: {e}")
        return

    # 2. Readiness Check
    try:
        resp = requests.get(f"{API_URL}/ready", timeout=5)
        if resp.status_code == 200:
            print(f"✅ /ready check: OK (Model: {resp.json().get('model')})")
        elif resp.status_code == 503:
            print("⚠️ /ready check: SERVICE UNAVAILABLE (Model loading or mock mode)")
        else:
            print(f"❌ /ready check FAILED with status {resp.status_code}")
    except Exception as e:
        print(f"⚠️ /ready check FAILED: {e} (Is the server running?)")

    # 3. Predict Endpoint
    create_dummy_video(TEST_VIDEO)
    try:
        with open(TEST_VIDEO, "rb") as f:
            files = {"file": (TEST_VIDEO, f, "video/mp4")}
            data = {"clip_id": "test_verification_001"}
            print(f"⏳ Sending prediction request to {API_URL}/predict...")
            resp = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=30)
            
        if resp.status_code == 200:
            result = resp.json()
            print("✅ /predict check: OK")
            print(f"   - Operation: {result['dominant_operation']}")
            print(f"   - Confidence: {result['confidence']}")
            print(f"   - Latency: {result['latency_ms']}ms")
        else:
            print(f"❌ /predict check FAILED: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ /predict check FAILED: {e}")
    finally:
        if os.path.exists(TEST_VIDEO):
            os.remove(TEST_VIDEO)

    print("\n🏁 Integration test sequence complete.")

if __name__ == "__main__":
    run_test()
