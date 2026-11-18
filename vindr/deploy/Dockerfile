# Public CPU demo for Fly.io. Build context = repo root:
#   fly deploy . --config deploy/fly.toml --dockerfile deploy/Dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/mpl
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY deploy/requirements.txt .
RUN pip install -r requirements.txt
COPY app.py features.py ./
COPY deploy/models ./models
EXPOSE 7860
CMD ["python", "app.py", "--public", "--device", "cpu", "--models_dir", "models", "--port", "7860"]
