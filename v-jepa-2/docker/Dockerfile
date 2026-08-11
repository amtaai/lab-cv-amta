# Imagen unica para todas las etapas del cascade (local, CPU).
# El host corre Python 3.14 externally-managed sin pip: todo Python vive aca.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        opencv-python-headless==4.10.0.84 \
        numpy==2.1.3 \
        pytest==8.3.3

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
