#!/usr/bin/env python
"""
LatentSync Paperspace Wrapper (GCS Version)

This script provides a RESTful API for LatentSync lip-syncing using GCP Storage:
1. Accept GCS paths for video and audio input
2. Process them with LatentSync
3. Output results to a specified GCS path
4. Provide a simple job-based interface for tracking

Usage:
  POST /jobs - Submit GCS paths and start processing
  GET /jobs/{job_id} - Check job status
  GET /jobs/{job_id}/log - View processing logs
"""

import os
import uuid
import time
import json
import shutil
import logging
import tempfile
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "/app/checkpoints"))
UNET_PATH = WEIGHTS_DIR / "latentsync_unet.pt"
WHISPER_PATH = WEIGHTS_DIR / "whisper/tiny.pt"
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/app/configs/unet/stage2.yaml"))

# Setup directories
JOBS_DIR = DATA_DIR / "jobs"
LOGS_DIR = DATA_DIR / "logs"

for directory in [JOBS_DIR, LOGS_DIR, WEIGHTS_DIR / "whisper"]:
    directory.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="LatentSync API (GCS Version)",
    description="API for processing videos with LatentSync lip-sync using Google Cloud Storage",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model for GCS paths
class GcsJobRequest(BaseModel):
    video_in: str = Field(..., description="URL to input video (http:// or gs://)")
    audio_in: str = Field(..., description="URL to input audio (http:// or gs://)")
    out: str = Field(..., description="URL for output video (http:// or gs://)")
    guidance_scale: float = Field(2.0, description="Guidance scale (1.0-3.0)")
    inference_steps: int = Field(20, description="Number of inference steps (10-50)")
    seed: int = Field(0, description="Random seed (0 for random)")

@app.post("/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    job_request: GcsJobRequest = Body(...)
):
    """
    Create a new lip-sync job using URLs
    
    - **video_in**: URL to input video (http:// or gs://)
    - **audio_in**: URL to input audio (http:// or gs://)
    - **out**: URL for output video (http:// or gs://)
    - **guidance_scale**: Guidance scale parameter (1.0-3.0)
    - **inference_steps**: Number of inference steps (10-50)
    - **seed**: Random seed (0 for random)
    
    Returns job details with URL for status checking
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    logger.info(f"New job received: {job_id}")
    
    # Create job directory
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Validate URLs
        for path_name, path_value in {
            "video_in": job_request.video_in,
            "audio_in": job_request.audio_in,
            "out": job_request.out
        }.items():
            if not (path_value.startswith("http://") or path_value.startswith("https://") or path_value.startswith("gs://")):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid URL for {path_name}: {path_value}. Must start with http://, https://, or gs://"
                )
        
        # Initialize job status
        write_job_status(job_id, "processing", params={
            "video_in": job_request.video_in,
            "audio_in": job_request.audio_in,
            "out": job_request.out,
            "guidance_scale": job_request.guidance_scale,
            "inference_steps": job_request.inference_steps,
            "seed": job_request.seed,
            "created_at": datetime.now().isoformat()
        })
        
        # Queue processing job
        background_tasks.add_task(
            process_gcs_job,
            job_id,
            job_request.video_in,
            job_request.audio_in,
            job_request.out,
            job_request.guidance_scale,
            job_request.inference_steps,
            job_request.seed
        )
        
        logger.info(f"Job {job_id} queued for processing")
        
        # Extract file names from URLs
        video_filename = Path(job_request.video_in).name
        audio_filename = Path(job_request.audio_in).name
        
        # Return job information
        return {
            "job_id": job_id,
            "status": "processing",
            "urls": {
                "video_in": job_request.video_in,
                "audio_in": job_request.audio_in,
                "out": job_request.out
            },
            "parameters": {
                "guidance_scale": job_request.guidance_scale,
                "inference_steps": job_request.inference_steps,
                "seed": job_request.seed
            },
            "created_at": datetime.now().isoformat(),
            "_links": {
                "self": f"/jobs/{job_id}",
                "log": f"/jobs/{job_id}/log"
            }
        }
    
    except Exception as e:
        logger.error(f"Error setting up job {job_id}: {str(e)}")
        # Clean up
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def process_gcs_job(
    job_id: str,
    video_in: str,
    audio_in: str,
    out_path: str,
    guidance_scale: float,
    inference_steps: int,
    seed: int
):
    """Process a job with LatentSync using URLs"""
    logger.info(f"Starting processing for job {job_id}")
    job_dir = JOBS_DIR / job_id
    log_path = LOGS_DIR / f"{job_id}.log"
    
    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        video_path = temp_dir_path / "input.mp4"
        audio_path = temp_dir_path / "input.wav"
        output_path = temp_dir_path / "result.mp4"
        
        try:
            # Create log file
            with open(log_path, "w") as f:
                f.write(f"Job {job_id} started at: {datetime.now().isoformat()}\n")
                f.write(f"Parameters:\n")
                f.write(f"  - video_in: {video_in}\n")
                f.write(f"  - audio_in: {audio_in}\n")
                f.write(f"  - out: {out_path}\n")
                f.write(f"  - guidance_scale: {guidance_scale}\n")
                f.write(f"  - inference_steps: {inference_steps}\n")
                f.write(f"  - seed: {seed}\n")
                f.write(f"  - model: {UNET_PATH}\n")
                f.write(f"  - config: {CONFIG_PATH}\n")
            
            # Download files
            with open(log_path, "a") as f:
                f.write("\n=== Downloading input files ===\n")
                
                # Download video
                f.write(f"Downloading video from {video_in}\n")
                try:
                    if video_in.startswith(("http://", "https://")):
                        # Use requests for http/https URLs
                        response = requests.get(video_in, stream=True)
                        response.raise_for_status()  # Raise an exception for HTTP errors
                        
                        with open(video_path, 'wb') as vf:
                            for chunk in response.iter_content(chunk_size=8192):
                                vf.write(chunk)
                    else:
                        # Unsupported gs:// URLs without gsutil
                        raise Exception("gs:// URLs are not supported. Please use HTTP(S) signed URLs.")
                    
                    f.write(f"Downloaded video to {video_path}\n")
                except Exception as e:
                    error_msg = f"Error downloading video: {str(e)}"
                    f.write(error_msg + "\n")
                    logger.error(f"Error downloading video for job {job_id}: {str(e)}")
                    raise Exception(error_msg)
                
                # Download audio
                f.write(f"Downloading audio from {audio_in}\n")
                try:
                    if audio_in.startswith(("http://", "https://")):
                        # Use requests for http/https URLs
                        response = requests.get(audio_in, stream=True)
                        response.raise_for_status()  # Raise an exception for HTTP errors
                        
                        with open(audio_path, 'wb') as af:
                            for chunk in response.iter_content(chunk_size=8192):
                                af.write(chunk)
                    else:
                        # Unsupported gs:// URLs without gsutil
                        raise Exception("gs:// URLs are not supported. Please use HTTP(S) signed URLs.")
                    
                    f.write(f"Downloaded audio to {audio_path}\n")
                except Exception as e:
                    error_msg = f"Error downloading audio: {str(e)}"
                    f.write(error_msg + "\n")
                    logger.error(f"Error downloading audio for job {job_id}: {str(e)}")
                    raise Exception(error_msg)
                
                f.write("Files downloaded successfully\n")
            
            # Check if model weights exist
            if not UNET_PATH.exists():
                raise FileNotFoundError(f"Model weights not found at {UNET_PATH}")
            
            if not WHISPER_PATH.exists():
                raise FileNotFoundError(f"Whisper model not found at {WHISPER_PATH}")
            
            start_time = time.time()
            
            # Run LatentSync inference
            cmd = [
                "python", "-m", "scripts.inference",
                "--unet_config_path", str(CONFIG_PATH),
                "--inference_ckpt_path", str(UNET_PATH),
                "--guidance_scale", str(guidance_scale),
                "--video_path", str(video_path),
                "--audio_path", str(audio_path),
                "--video_out_path", str(output_path),
                "--seed", str(seed),
                "--inference_steps", str(inference_steps)
            ]
            
            logger.info(f"Running LatentSync for job {job_id}")
            
            # Run LatentSync and capture output
            with open(log_path, "a") as f:
                f.write("\n=== Running LatentSync ===\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                
                try:
                    process = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=True
                    )
                    f.write("LatentSync output:\n")
                    f.write(process.stdout)
                    logger.info(f"LatentSync process completed successfully for job {job_id}")
                except subprocess.CalledProcessError as e:
                    f.write(f"LatentSync failed with exit code {e.returncode}:\n")
                    f.write(e.stdout)
                    logger.error(f"LatentSync process failed for job {job_id} with exit code {e.returncode}")
                    raise Exception(f"LatentSync processing failed with exit code {e.returncode}")
            
            # Verify output file exists
            if not output_path.exists():
                raise FileNotFoundError(f"Output file not created")
            
            # Upload result
            with open(log_path, "a") as f:
                f.write("\n=== Uploading result ===\n")
                f.write(f"Uploading to {out_path}\n")
                
                try:
                    if out_path.startswith(("http://", "https://")):
                        # Upload using requests for HTTP(S) URLs
                        with open(output_path, 'rb') as out_file:
                            headers = {'Content-Type': 'video/mp4'}
                            response = requests.put(out_path, data=out_file, headers=headers)
                            response.raise_for_status()
                    else:
                        # Unsupported gs:// URLs without gsutil
                        raise Exception("gs:// URLs are not supported for upload. Please use HTTP(S) signed URLs.")
                    
                    f.write("Upload completed successfully\n")
                except Exception as e:
                    error_msg = f"Error uploading result: {str(e)}"
                    f.write(error_msg + "\n")
                    logger.error(f"Error uploading result for job {job_id}: {str(e)}")
                    raise Exception(error_msg)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update job status to completed
            write_job_status(job_id, "completed", metadata={
                "processing_time": processing_time,
                "completed_at": datetime.now().isoformat()
            })
            
            logger.info(f"Job {job_id} completed successfully in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            write_job_status(job_id, "failed", error=str(e))
            
            # Log error
            with open(log_path, "a") as f:
                f.write(f"\nERROR: {str(e)}\n")

def write_job_status(
    job_id: str, 
    status: str, 
    error: Optional[str] = None, 
    metadata: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None
):
    """Write job status to a JSON file"""
    status_file = JOBS_DIR / job_id / "status.json"
    
    # Read existing status if available
    if status_file.exists():
        with open(status_file, "r") as f:
            status_data = json.load(f)
    else:
        status_data = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat()
        }
    
    # Update status
    status_data["status"] = status
    status_data["updated_at"] = datetime.now().isoformat()
    
    # Add optional fields
    if error:
        status_data["error"] = error
    
    if metadata:
        if "metadata" not in status_data:
            status_data["metadata"] = {}
        status_data["metadata"].update(metadata)
    
    if params:
        if "parameters" not in status_data:
            status_data["parameters"] = {}
        status_data["parameters"].update(params)
    
    # Add links
    status_data["_links"] = {
        "self": f"/jobs/{job_id}",
        "log": f"/jobs/{job_id}/log"
    }
    
    # Write status file
    with open(status_file, "w") as f:
        json.dump(status_data, f, indent=2)

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Get job status and details
    
    - **job_id**: ID of the job to retrieve
    
    Returns complete job information including status and URL (if completed)
    """
    # Check if job directory exists
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Get status file
    status_file = job_dir / "status.json"
    if not status_file.exists():
        return {
            "job_id": job_id,
            "status": "unknown",
            "_links": {
                "self": f"/jobs/{job_id}"
            }
        }
    
    # Read status data
    with open(status_file, "r") as f:
        status_data = json.load(f)
    
    return status_data

@app.get("/jobs/{job_id}/log")
async def get_job_log(job_id: str):
    """
    Get processing logs for a job
    
    - **job_id**: ID of the job
    
    Returns the log file containing processing details and any errors
    """
    # Check if job exists
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Check if log file exists
    log_file = LOGS_DIR / f"{job_id}.log"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    
    # Return the log file
    return FileResponse(
        path=log_file,
        media_type="text/plain",
        filename=f"latentsync_{job_id}.log"
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns service health status and configuration details
    """
    # Check if weights exist
    weights_exist = UNET_PATH.exists() and WHISPER_PATH.exists()
    config_exists = CONFIG_PATH.exists()
    
    # No need to check for gsutil as we're using requests
    status = "healthy" if weights_exist and config_exists else "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "unet_weights": str(UNET_PATH.exists()),
            "whisper_weights": str(WHISPER_PATH.exists()),
            "config": str(CONFIG_PATH.exists()),
            "data_dir": str(DATA_DIR.exists())
        }
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "LatentSync API (HTTP URLs Version)",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "/jobs": "POST - Create a new lip-sync job using URLs",
            "/jobs/{job_id}": "GET - Check job status",
            "/jobs/{job_id}/log": "GET - View processing logs",
            "/health": "GET - Service health check"
        }
    }

if __name__ == "__main__":
    # When run directly, start the uvicorn server
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting LatentSync API server on port {port}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Weights directory: {WEIGHTS_DIR}")
    logger.info(f"Config path: {CONFIG_PATH}")
    
    # Check for model weights
    if not UNET_PATH.exists():
        logger.warning(f"LatentSync model weights not found at {UNET_PATH}")
    else:
        logger.info(f"Found LatentSync model weights at {UNET_PATH}")
    
    if not WHISPER_PATH.exists():
        logger.warning(f"Whisper model not found at {WHISPER_PATH}")
    else:
        logger.info(f"Found Whisper model at {WHISPER_PATH}")
    
    # Start server
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")