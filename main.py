#!/usr/bin/env python3

import os
import json
import tempfile
import subprocess
import logging
from typing import Dict, Any
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def process_video(video_path: str, audio_path: str, output_path: str, 
                  guidance_scale: float = 2.0, inference_steps: int = 20, 
                  seed: int = 0) -> None:
    config_path = "configs/unet/stage2.yaml"
    ckpt_path = "checkpoints/latentsync_unet.pt"

    if not os.path.exists(ckpt_path) or not os.path.exists("checkpoints/whisper/tiny.pt"):
        raise FileNotFoundError("Model checkpoints not found.")

    cmd = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", config_path,
        "--inference_ckpt_path", ckpt_path,
        "--guidance_scale", str(guidance_scale),
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_path,
        "--seed", str(seed),
        "--inference_steps", str(inference_steps)
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False, capture_output=False, text=True)
        logger.info(f"Command output: {result.stdout}")
        print(f"Command output: {result.stdout}")
        if result.stderr:
            logger.info(f"Command stderr: {result.stderr}")
            print(f"Command stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.info(f"Command failed: {e.stderr}")
        print(f"Command failed: {e.stderr}")
        raise

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file {output_path} not created")
    logger.info(f"Output saved to {output_path}")

def handle_job(job_data: Dict[str, Any]) -> Dict[str, Any]:
    required_fields = ["video_in", "audio_in", "out"]
    for field in required_fields:
        if field not in job_data:
            return {"error": f"Missing required field: {field}"}, 400

    video_in = job_data["video_in"]
    audio_in = job_data["audio_in"]
    out_path = job_data["out"]
    guidance_scale = float(job_data.get("guidance_scale", 2.0))
    inference_steps = int(job_data.get("inference_steps", 20))
    seed = int(job_data.get("seed", 0))

    try:
        process_video(
            video_path=video_in,
            audio_path=audio_in,
            output_path=out_path,
            guidance_scale=guidance_scale,
            inference_steps=inference_steps,
            seed=seed
        )

        return {
            "status": "success",
            "message": f"Processed video saved to {out_path}",
            "video_in": video_in,
            "audio_in": audio_in,
            "out": out_path
        }

    except Exception as e:
        logger.exception(f"Error processing job: {e}")
        return {"error": str(e), "status": "error"}, 500

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LatentSync local job processor")
    parser.add_argument("--video_in", required=True)
    parser.add_argument("--audio_in", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    job = {
        "video_in": args.video_in,
        "audio_in": args.audio_in,
        "out": args.out,
        "guidance_scale": args.guidance_scale,
        "inference_steps": args.inference_steps,
        "seed": args.seed
    }

    result = handle_job(job)
    if isinstance(result, tuple):
        print(f"Error: {result[0]['error']}")
        exit(1)
    print(json.dumps(result, indent=2))
