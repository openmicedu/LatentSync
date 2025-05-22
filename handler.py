"""
runpod/handler.py  –  concurrent, GPU-aware

Processes up to 3 jobs concurrently inside one worker.  The heavy
LatentSync call is run in a background thread (async-friendly).
"""

import runpod, asyncio, uuid, subprocess, shutil, pathlib
from functools import partial
from main import handle_job                     # ← your existing CPU/GPU logic

TMP = pathlib.Path("/tmp")
MAX_CONCURRENCY = 3                             # hard-cap per worker


# ---------- helper ----------------------------------------------------------
def _dl(url: str, dest: pathlib.Path):
    subprocess.check_call(["curl", "-L", "-sS", "-o", str(dest), url])

def _ul(src: pathlib.Path, url: str):
    subprocess.check_call(["curl", "-X", "PUT", "-T", str(src), url])


# ---------- async handler ---------------------------------------------------
async def process_request(job: dict):
    """
    RunPod passes {"id": "...", "input": {...}}.
    We do all blocking work in a thread to keep the event-loop free.
    """
    print(f"processing {job.get('id')}")
    job_input = job["input"]
    work = TMP / uuid.uuid4().hex
    work.mkdir(exist_ok=True)

    try:
        v_raw, a_raw, o_local = work/"in.mp4", work/"in.wav", work/"out.mp4"
        _dl(job_input["video"], v_raw)
        _dl(job_input["audio"], a_raw)

        # LatentSync is blocking → run in a thread
        fn = partial(handle_job, {
            "video_in": str(v_raw),
            "audio_in": str(a_raw),
            "out":      str(o_local),
            "inference_steps": job_input.get("steps", 20)
        })
        await asyncio.to_thread(fn)

        _ul(o_local, job_input["out"])
        return {"status": "done", "out": job_input["out"]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- dynamic concurrency (optional) ----------------------------------
def adjust_concurrency(current: int) -> int:
    """
    Keep at most MAX_CONCURRENCY requests in flight.
    You can add logic here (GPU util, queue backlog, etc.).
    """
    print("setting concurrency to 3")
    return 3


# ---------- start the worker ------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({
        "handler":               process_request,   # async handler
        "concurrency_modifier":  adjust_concurrency
    })
