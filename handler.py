# runpod/handler.py  ── concurrent, non-blocking

import asyncio
import subprocess
import uuid
import pathlib
import shutil
import runpod                     # RunPod serverless SDK
from main import handle_job       # your existing LatentSync wrapper

MAX_CONCURRENCY = 2               # how many requests per worker

# ──────────────────────────────────────────────────────────────
# Helper functions (run in background threads)
# ──────────────────────────────────────────────────────────────
def _dl(url: str, dest: pathlib.Path):
    """Download URL → dest (async via to_thread)."""
    subprocess.check_call(["curl", "-L", "-sS", "-o", str(dest), url])

def _ul(src: pathlib.Path, url: str):
    """Upload src → signed URL (async via to_thread)."""
    subprocess.check_call(["curl", "-X", "PUT", "-T", str(src), url])

# ──────────────────────────────────────────────────────────────
# Main async handler
# ──────────────────────────────────────────────────────────────
async def process_request(job: dict):
    """
    RunPod passes {"id": "...", "input": {...}}
    Expected fields in input:
        video  – signed GET URL (mp4)
        audio  – signed GET URL (wav)
        out    – signed PUT  URL (mp4)
        steps  – optional int, defaults to 20
    """
    inp  = job["input"]
    TMP_DIR = pathlib.Path("/tmp")
    work = TMP_DIR / uuid.uuid4().hex
    work.mkdir(exist_ok=True)

    try:
        v_local = work / "in.mp4"
        a_local = work / "in.wav"
        o_local = work / "out.mp4"

        _dl(inp["video"], v_local)
        _dl(inp["audio"], a_local)
        # 2. heavy LatentSync work on GPU (runs in a thread)
        await asyncio.to_thread(
            handle_job,
            {
                "video_in":      str(v_local),
                "audio_in":      str(a_local),
                "out":           str(o_local),
                "inference_steps": inp.get("steps", 20)
            }
        )

        # 3. upload result
        _ul(o_local, inp["out"])

        return {"status": "done", "out": inp["out"]}
    finally:
        shutil.rmtree(work, ignore_errors=True)

# ──────────────────────────────────────────────────────────────
# Optional dynamic modifier (keeps ≤ MAX_CONCURRENCY jobs)
# ──────────────────────────────────────────────────────────────
def concurrency_modifier(current: int) -> int:
    """RunPod calls this periodically to adjust live concurrency."""
    return MAX_CONCURRENCY
    
# ──────────────────────────────────────────────────────────────
# Start the worker
# ──────────────────────────────────────────────────────────────
runpod.serverless.start({
    "handler":               process_request,
    "concurrency_modifier":  concurrency_modifier
})