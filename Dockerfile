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
RUN pip install flask google-cloud-storage

RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
 && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && apt-get update && apt-get install -y google-cloud-sdk \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
 
ENTRYPOINT ["/bin/bash","-c", "\
  if [ ! -f /app/checkpoints/latentsync_unet.pt ]; then \
     mkdir -p /app/checkpoints && \
     gsutil -q cp -r gs://${PROJECT_ID}-latentsync-stage-latentsync-weights/checkpoints /app/ ; \
  fi && \
  python main.py \"$@\" "]