#!/usr/bin/env python3
"""
LatentSync GCP Cloud Run Wrapper

This script wraps the LatentSync predict.py functionality to:
1. Download input video and audio from GCS
2. Process them with LatentSync
3. Upload the result back to GCS
4. Support both direct HTTP requests and Pub/Sub messages

Usage:
  HTTP POST: /process
    JSON body: {"video_in": "gs://bucket/video.mp4", "audio_in": "gs://bucket/audio.wav", "out": "gs://bucket/output.mp4"}
  
  Pub/Sub message with same JSON format
"""

import os
import json
import tempfile
import argparse
import subprocess
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import urllib.parse

import flask
from flask import Flask, request, jsonify
from google.cloud import storage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize GCS client
storage_client = storage.Client()

def parse_gcs_path(gcs_path: str) -> Tuple[str, str]:
    """
    Parses a GCS path into bucket name and blob path.
    
    Args:
        gcs_path: Path in format gs://bucket-name/path/to/file
        
    Returns:
        Tuple of (bucket_name, blob_path)
    """
    parsed_url = urllib.parse.urlparse(gcs_path)
    if parsed_url.scheme != "gs":
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with gs://")
    
    bucket_name = parsed_url.netloc
    blob_path = parsed_url.path.lstrip('/')
    return bucket_name, blob_path

def download_from_gcs(gcs_path: str, local_path: str) -> None:
    """
    Downloads a file from GCS to a local path.
    
    Args:
        gcs_path: Path in format gs://bucket-name/path/to/file
        local_path: Local file path to save the downloaded file
    """
    bucket_name, blob_path = parse_gcs_path(gcs_path)
    
    logger.info(f"Downloading from gs://{bucket_name}/{blob_path} to {local_path}")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded {gcs_path}")

def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """
    Uploads a file from a local path to GCS.
    
    Args:
        local_path: Local file path to upload
        gcs_path: Path in format gs://bucket-name/path/to/file
    """
    bucket_name, blob_path = parse_gcs_path(gcs_path)
    
    logger.info(f"Uploading from {local_path} to gs://{bucket_name}/{blob_path}")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded to {gcs_path}")

def process_video(video_path: str, audio_path: str, output_path: str, 
                 guidance_scale: float = 2.0, inference_steps: int = 20, 
                 seed: int = 0) -> None:
    """
    Process a video using LatentSync.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to input audio file
        output_path: Path to save the output video
        guidance_scale: Guidance scale parameter (1.0-3.0)
        inference_steps: Number of inference steps (10-50)
        seed: Random seed (0 for random)
    """
    config_path = "configs/unet/stage2.yaml"
    ckpt_path = "checkpoints/latentsync_unet.pt"
    
    # Ensure the checkpoints directory exists
    if not os.path.exists("checkpoints/latentsync_unet.pt") or not os.path.exists("checkpoints/whisper/tiny.pt"):
        raise FileNotFoundError("Model checkpoints not found. Please check if they've been downloaded correctly.")
    
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Command output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"Command stdout: {e.stdout}")
        logger.error(f"Command stderr: {e.stderr}")
        raise
    
    # Verify the output file exists
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file {output_path} was not created")
    
    logger.info(f"Processing complete. Output saved to {output_path}")

def handle_job(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a LatentSync job from either HTTP or Pub/Sub.
    
    Args:
        job_data: Dictionary containing job parameters
        
    Returns:
        Dictionary with job results
    """
    required_fields = ["video_in", "audio_in", "out"]
    for field in required_fields:
        if field not in job_data:
            return {"error": f"Missing required field: {field}"}, 400
    
    # Extract parameters
    video_in = job_data["video_in"]
    audio_in = job_data["audio_in"]
    out_path = job_data["out"]
    guidance_scale = float(job_data.get("guidance_scale", 2.0))
    inference_steps = int(job_data.get("inference_steps", 20))
    seed = int(job_data.get("seed", 0))
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Download input files
        video_filename = Path(parse_gcs_path(video_in)[1]).name
        audio_filename = Path(parse_gcs_path(audio_in)[1]).name
        output_filename = Path(parse_gcs_path(out_path)[1]).name
        
        local_video_path = str(temp_dir_path / video_filename)
        local_audio_path = str(temp_dir_path / audio_filename)
        local_output_path = str(temp_dir_path / output_filename)
        
        try:
            download_from_gcs(video_in, local_video_path)
            download_from_gcs(audio_in, local_audio_path)
            
            # Process video
            process_video(
                video_path=local_video_path,
                audio_path=local_audio_path,
                output_path=local_output_path,
                guidance_scale=guidance_scale,
                inference_steps=inference_steps,
                seed=seed
            )
            
            # Upload result
            upload_to_gcs(local_output_path, out_path)
            
            return {
                "status": "success",
                "message": f"Processed video uploaded to {out_path}",
                "video_in": video_in,
                "audio_in": audio_in,
                "out": out_path
            }
            
        except Exception as e:
            logger.exception(f"Error processing job: {e}")
            return {"error": str(e), "status": "error"}, 500

@app.route("/", methods=["GET"])
def home():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "LatentSync Video Processor",
        "version": "1.0.0"
    })

@app.route("/process", methods=["POST"])
def process_http_request():
    """HTTP endpoint for processing a video"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    try:
        job_data = request.get_json()
        result = handle_job(job_data)
        
        # If result is a tuple, it's an error with status code
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        
        return jsonify(result)
        
    except Exception as e:
        logger.exception(f"Error handling HTTP request: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/pubsub", methods=["POST"])
def process_pubsub_message():
    """Pub/Sub push subscription endpoint"""
    try:
        envelope = request.get_json()
        
        if not envelope:
            return jsonify({"error": "No Pub/Sub message received"}), 400
            
        if not isinstance(envelope, dict) or "message" not in envelope:
            return jsonify({"error": "Invalid Pub/Sub message format"}), 400
            
        # Extract the message
        pubsub_message = envelope["message"]
        
        if "data" not in pubsub_message:
            return jsonify({"error": "No data in Pub/Sub message"}), 400
            
        # Decode the Pub/Sub data from base64
        import base64
        data_str = base64.b64decode(pubsub_message["data"]).decode("utf-8")
        job_data = json.loads(data_str)
        
        # Process the job
        result = handle_job(job_data)
        
        # If result is a tuple, it's an error with status code
        if isinstance(result, tuple):
            # For Pub/Sub, we should return 200 OK even for job errors
            # to acknowledge receipt (prevent redelivery)
            logger.error(f"Job processing error: {result[0]}")
            return jsonify({"status": "error", "message": result[0]["error"]}), 200
        
        return jsonify({"status": "success"})
        
    except Exception as e:
        logger.exception(f"Error handling Pub/Sub message: {e}")
        # For Pub/Sub, we should return 200 OK for exceptions to acknowledge receipt
        return jsonify({"status": "error", "message": str(e)}), 200

def main():
    """Command-line interface for direct invocation"""
    parser = argparse.ArgumentParser(description="LatentSync GCP Cloud Run Wrapper")
    parser.add_argument("--video_in", type=str, required=True, help="GCS path to input video (gs://...)")
    parser.add_argument("--audio_in", type=str, required=True, help="GCS path to input audio (gs://...)")
    parser.add_argument("--out", type=str, required=True, help="GCS path for output video (gs://...)")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Guidance scale (1.0-3.0)")
    parser.add_argument("--inference_steps", type=int, default=20, help="Number of inference steps (10-50)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random)")
    
    args = parser.parse_args()
    
    job_data = {
        "video_in": args.video_in,
        "audio_in": args.audio_in,
        "out": args.out,
        "guidance_scale": args.guidance_scale,
        "inference_steps": args.inference_steps,
        "seed": args.seed
    }
    
    result = handle_job(job_data)
    
    if isinstance(result, tuple):
        print(f"Error: {result[0]['error']}")
        exit(1)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    # Check if running as CLI or as web service
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Get port from environment variable or default to 8080
        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port, debug=False)