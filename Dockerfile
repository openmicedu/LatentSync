FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive

# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        build-essential        \  
        git                    \
        cmake                  \
        ninja-build            \
        libopenblas-dev        \
        liblapack-dev          \
        libopencv-dev          \
        ffmpeg curl wget libgl1-mesa-glx libglib2.0-0 ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3  /usr/bin/pip

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app/checkpoints
RUN mkdir -p whisper && \
    wget -q --show-progress -O latentsync_unet.pt \
        https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/latentsync_unet.pt && \
    wget -q --show-progress -O stable_syncnet.pt \
        https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/stable_syncnet.pt && \
    wget -q --show-progress -O whisper/tiny.pt \
        https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/whisper/tiny.pt

# copy source and (optionally) checkpoints
COPY . .

CMD ["python", "-u", "handler.py"]
