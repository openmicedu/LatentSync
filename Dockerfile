FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg curl libgl1-mesa-glx libglib2.0-0        && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source and (optionally) checkpoints
COPY . .
# COPY checkpoints/ checkpoints/

CMD ["python", "-u", "handler.py"]
