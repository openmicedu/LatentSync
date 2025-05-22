FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0 python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .
RUN pip install fastapi==0.115.12 uvicorn[standard]==0.34.2 python-multipart==0.0.20
RUN mkdir -p /app/data/jobs /app/data/logs /app/checkpoints/whisper


ENV DATA_DIR=/app/data
ENV WEIGHTS_DIR=/app/checkpoints
ENV CONFIG_PATH=/app/configs/unet/stage2.yaml
ENV PORT=8080
EXPOSE 8080
CMD ["python", "main.py"]
