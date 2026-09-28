# Imagen unica para todas las etapas del cascade.
# El host corre Python 3.14 externally-managed sin pip: todo Python vive aca.
#
# El Nivel 2 corre en la GPU (RTX 2060). Las wheels cu121 de torch traen su
# propio runtime CUDA, asi que no hace falta una imagen base nvidia/cuda: alcanza
# con que el host tenga driver + nvidia-container-toolkit y que el compose pida
# el device. Turing (sm_75) esta soportado por estas wheels.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch primero y desde el indice cu121: si se deja que ultralytics lo resuelva,
# pip baja la build de CPU y la GPU queda sin usar sin avisar.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1

# ultralytics trae YOLO11-seg, YOLO-World y ByteTrack: los mismos modelos y el
# mismo tracker que ya usan los notebooks de yolo_seg/ y yolo_world/.
#
# transformers va PINNEADO a 4.46.3, no a la ultima: desde 4.47 exige torch>=2.5 y
# subir torch cambiaria los ms/frame ya medidos del detector. 4.46.3 soporta DINOv2
# de sobra (esta desde 4.31).
#
# lapx y shapely van explicitos: el primero lo necesita ByteTrack para la
# asignacion lineal y el segundo ultralytics.solutions para la geometria de las
# zonas. Si faltan, ultralytics intenta instalarlos EN RUNTIME con pip, y eso
# revienta cuando el contenedor corre como usuario no-root (que es como hay que
# correrlo para no dejar archivos de root en el host).
RUN pip install --no-cache-dir \
        ultralytics==8.3.40 \
        lapx==0.5.11 \
        shapely==2.1.2 \
        transformers==4.46.3 \
        opencv-python-headless==4.10.0.84 \
        pytest==8.3.3

# La API REST. httpx lo necesita TestClient de starlette para correr los tests sin
# levantar un servidor, y python-multipart el endpoint de subida de video.
RUN pip install --no-cache-dir \
        fastapi==0.115.6 \
        "uvicorn[standard]==0.34.0" \
        python-multipart==0.0.20 \
        httpx==0.28.1

# Los pesos van EN LA IMAGEN, no en el volumen: asi la corrida no depende de la
# red y el contenedor no necesita escribir en /app para funcionar.
ENV AMTA_MODELS_DIR=/opt/amta/models
RUN mkdir -p /opt/amta/models && cd /opt/amta/models && \
    python -c "\
from ultralytics.utils.downloads import attempt_download_asset;\
[attempt_download_asset(n) for n in ('yolo11n-seg.pt', 'yolov8s-worldv2.pt')]" && \
    chmod -R a+rX /opt/amta/models && ls -la /opt/amta/models

# El extractor de apariencia del Nivel 3 (ReID), tambien en la imagen.
# DINOv2 y no DINOv3: el v3 esta gated en HuggingFace (`gated: manual`, hay que
# pedirle acceso a Meta y que lo aprueben). Misma familia y misma API AutoModel,
# asi que cuando haya acceso se cambia el nombre del modelo y nada mas.
ENV HF_HOME=/opt/amta/hf
RUN HF_HOME=/opt/amta/hf python -c "\
from transformers import AutoImageProcessor, AutoModel;\
m='facebook/dinov2-small';\
AutoImageProcessor.from_pretrained(m); AutoModel.from_pretrained(m)" && \
    chmod -R a+rX /opt/amta/hf

# HOME y las caches tienen que ser escribibles por cualquier UID: el contenedor
# corre con --user $(id -u) para no dejar archivos de root en el host, y sin esto
# matplotlib y fontconfig llenan la salida de errores de permisos.
ENV HOME=/tmp
ENV MPLCONFIGDIR=/tmp/mpl
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
