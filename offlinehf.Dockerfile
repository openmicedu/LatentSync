FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg curl libgl1-mesa-glx libglib2.0-0        && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app
COPY checkpoints/ checkpoints/
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install openai-whisper==20240930 soundfile==0.12.1

# copy source and (optionally) checkpoints
COPY . .

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/app/checkpoints \ 
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DIFFUSERS_OFFLINE=1 \
    TORCH_HOME=/app/checkpoints \
    INSIGHTFACE_HOME=/app/checkpoints/insightface

RUN ln -s /app/checkpoints/stabilityai /app/stabilityai

CMD ["python", "-u", "handler.py"]
