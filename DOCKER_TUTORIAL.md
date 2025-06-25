# build online image
```shell
docker build -t us-central1-docker.pkg.dev/openmic-backend-stg/latentsync/worker:runpod --platform linux/amd64 .
```

# build offline image
```shell
gsutil cp gs://openmic-devops/artifacts/checkpoints.zip
unzip checkpoints.zip
docker build -t us-central1-docker.pkg.dev/openmic-backend-stg/latentsync/worker:runpod --platform linux/amd64 -f offline.Dockerfile .
```
