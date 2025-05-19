FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu124.py310

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0 python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git  \
        ffmpeg libavcodec-dev libavfilter-dev \
        libavformat-dev libavutil-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .
RUN pip install flask google-cloud-storage

ENTRYPOINT ["/bin/bash","-c", "\
  if [ ! -f /app/checkpoints/latentsync_unet.pt ]; then \
     mkdir -p /app/checkpoints && \
     gsutil -q cp -r gs://${PROJECT_ID}-latentsync-stage-latentsync-weights/checkpoints /app/ ; \
  fi && \
  python main.py \"$@\" "]
