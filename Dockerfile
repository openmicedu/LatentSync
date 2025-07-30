FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg curl libgl1-mesa-glx libglib2.0-0        && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install -y python3 python3-pip && wget \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app/checkpoints
RUN wget -O latentsync_unet.pt "https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/latentsync_unet.pt"
RUN wget -O stable_syncnet.pt "https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/stable_syncnet.pt"
RUN wget -O whisper/tiny.pt "https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/whisper/tiny.pt"

COPY requirements.txt .
RUN pip install -r requirements.txt

# copy source and (optionally) checkpoints
COPY . .

CMD ["python", "-u", "handler.py"]
