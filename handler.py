import runpod, uuid, subprocess, shutil, pathlib
from main import handle_job

TMP = pathlib.Path("/tmp")

def download(url, dest):
    subprocess.check_call(["curl", "-L", "-sS", "-o", dest, url])

def upload(src, url):
    subprocess.check_call(["curl", "-X", "PUT", "-T", src, url])

def handler(event):
    job = event["input"]
    work = TMP / uuid.uuid4().hex
    work.mkdir(exist_ok=True)
    try:
        v_local = work / "in.mp4"
        a_local = work / "in.wav"
        o_local = work / "out.mp4"

        download(job["video"], v_local)
        download(job["audio"], a_local)

        handle_job({
            "video_in": str(v_local),
            "audio_in": str(a_local),
            "out":      str(o_local),
            "inference_steps":    job.get("steps", 20)
        })

        upload(o_local, job["out"])
        return {"status": "done", "out": job["out"]}
    finally:
        shutil.rmtree(work, ignore_errors=True)

runpod.serverless.start({"handler": handler})