FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# ---- system deps ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \       
        build-essential \
        git cmake ninja-build \
        libopenblas-dev liblapack-dev libopencv-dev \
        ffmpeg curl wget libgl1-mesa-glx libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3   /usr/bin/pip

# --------------------------------------------------------------------------

WORKDIR /app/checkpoints
RUN mkdir -p whisper && \
    wget -q --show-progress -O latentsync_unet.pt \
        https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/latentsync_unet.pt && \
    wget -q --show-progress -O stable_syncnet.pt \
        https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/stable_syncnet.pt && \
    wget -q --show-progress -O whisper/tiny.pt \
        https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/whisper/tiny.pt


WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip         \
 && python -m pip install --no-cache-dir -r requirements.txt



# copy source and (optionally) checkpoints
COPY . .

CMD ["python", "-u", "handler.py"]
