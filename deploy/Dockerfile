# Public CPU demo (Fly.io). Build context = repo root:
#   fly deploy . --config deploy/fly.toml --dockerfile deploy/Dockerfile --remote-only --ha=false
# Python version matches the training environment (the logistic-regression model is a pickle).
FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/mpl GRADIO_ANALYTICS_ENABLED=False
WORKDIR /app
COPY deploy/requirements.txt .
RUN pip install -r requirements.txt
COPY acdc ./acdc
COPY deploy/models ./models
EXPOSE 7860
CMD ["python", "-m", "acdc.app", "--public", "--device", "cpu", "--models_dir", "models", "--port", "7860"]
