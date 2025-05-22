# runpod/handler.py  ── concurrent, non-blocking

import asyncio
import runpod                     # RunPod serverless SDK

# ──────────────────────────────────────────────────────────────
# Dummy request handler
# ──────────────────────────────────────────────────────────────
async def process_request_dummy(job: dict):
    print(f"process_request start, job id: {job['id']}")
    await asyncio.sleep(60)
    return {"status": "done", "out": "finished"}

# ──────────────────────────────────────────────────────────────
# Optional dynamic modifier (keeps ≤ MAX_CONCURRENCY jobs)
# ──────────────────────────────────────────────────────────────
def concurrency_modifier(current: int) -> int:
    """RunPod calls this periodically to adjust live concurrency."""
    print("start concurrency_modifier")
    return 3

# ──────────────────────────────────────────────────────────────
# Start the worker
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("start concurrency_main")
    runpod.serverless.start({
        "handler":               process_request_dummy,
        "concurrency_modifier":  concurrency_modifier
    })
