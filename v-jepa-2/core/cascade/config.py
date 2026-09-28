"""Configuracion del cascade de vigilancia (Semana 1: simulador, corpus, motion, costo).

Topologia: TODO el Python corre dentro del contenedor `cascade` (python:3.12-slim).
El host tiene Python 3.14 externally-managed sin pip — ver docker/Dockerfile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = ROOT / "corpus"  # indice + manifest versionados; raw/ gitignoreado
RAW_DIR = CORPUS_DIR / "raw"  # los videos en si, nunca versionados
RESULTS_DIR = ROOT / "results"  # stats json + reporte md versionados; costs.db no
MODELS_DIR = ROOT / "models"  # pesos de los modelos, nunca versionados


@dataclass
class MotionConfig:
    """Parametros del Nivel 1 (MOG2). Fijados a ojo sobre UNA secuencia de muestra."""

    history: int = 500  # frames que MOG2 recuerda para el modelo de fondo
    var_threshold: float = 16.0  # umbral de Mahalanobis del pixel vs el modelo
    detect_shadows: bool = True  # sombras salen como 127, se filtran del foreground
    # Area minima del contorno para contar como movimiento, como FRACCION del
    # frame. Tiene que ser relativa: con un umbral absoluto en pixeles, el mismo
    # valor es 6.75x mas sensible a 1080p que a 480p y un corpus de resolucion
    # mixta da resultados incomparables (medido: 99.9% vs 62.1% de movimiento).
    # 0.0016 = 500 px en 640x480, el valor absoluto que se usaba antes.
    min_area_frac: float = 0.0016
    warmup_frames: int = 30  # MOG2 marca todo como foreground al arrancar: excluir
    flicker_fg_ratio: float = 0.50  # foreground por encima de esto = cambio de luz global
    motion_fg_ratio_min: float = 0.0005  # piso: menos que esto es ruido de sensor


@dataclass
class PersonConfig:
    """Parametros del Nivel 2.

    El detector NO vive aca: es el de `core/perception/detector.py`, el mismo que
    usan los notebooks de `yolo_seg/` y `yolo_world/`. Esto es solo como lo
    configura el cascade.
    """

    # "yolo11" = clases COCO fijas, rapido (yolo_seg/). "yoloworld" = open-vocab
    # por texto (yolo_world/). Los dos corren ByteTrack adentro de ultralytics.
    backend: str = os.environ.get("AMTA_DETECTOR", "yolo11")
    pesos: str = os.environ.get("AMTA_PESOS", "")  # vacio = el default del backend
    # Lo que se le pide al detector. Con yoloworld es literalmente el prompt; con
    # yolo11 se traduce al indice de la clase COCO que coincida por nombre.
    prompts: tuple[str, ...] = ("person",)
    imgsz: int = int(os.environ.get("AMTA_IMGSZ", "640"))
    conf: float = float(os.environ.get("AMTA_CONF", "0.35"))
    iou: float = float(os.environ.get("AMTA_IOU", "0.70"))
    # "0" = primera GPU, "cpu" = sin GPU. El Nivel 2 esta pensado para GPU.
    device: str = os.environ.get("AMTA_DEVICE", "0")


@dataclass
class TrackConfig:
    """Parametros del registro de tracks (core/perception/tracker.py).

    La asociacion la hace ByteTrack adentro de ultralytics; esto solo gobierna
    cuando un track cuenta como una persona real.
    """

    tracker: str = os.environ.get("AMTA_TRACKER", "bytetrack.yaml")  # o botsort.yaml
    min_hits: int = 3  # detecciones antes de contarlo como instancia
    max_age: int = 30  # frames sin verse antes de darlo por ido


@dataclass
class ReidCascadeConfig:
    """Nivel 3: re-identificacion por apariencia (core/perception/reid.py).

    Corre DESPUES de ByteTrack y solo corrige sus cortes. Medido sobre los clips
    de tienda: 207 IDs para un puñado de personas reales bajan a 49, y en el clip
    dificil de 134 a 28 contra ~20 personas de verdad.

    Se puede apagar con AMTA_REID=0: el resto del cascade sigue funcionando igual,
    solo que los IDs se parten con cada oclusion.
    """

    activo: bool = os.environ.get("AMTA_REID", "1") not in ("0", "false", "no")


@dataclass
class ConteoConfig:
    """Linea de conteo (punto 3).

    En coordenadas normalizadas [0,1] del frame, nunca en pixeles: el corpus
    mezcla 640x480 con 1920x1080 y una geometria en pixeles caeria en lugares
    distintos en cada resolucion. Es el mismo error que ya se corrigio en el
    umbral de area del Nivel 1.

    Linea vertical: quien la cruza hacia la derecha "entra" y hacia la izquierda
    "sale". x=0.35 NO es una eleccion de diseno, es donde este corpus tiene
    trafico. Se midieron 31 trayectorias sobre 10 clips: la gente vive entre
    x=0.05 y x=0.51 (p5-p95) y ninguna llega a la mitad del cuadro, asi que una
    linea al centro daba CERO cruces. Cruces por posicion: x=0.25 -> 3,
    x=0.30 -> 7, x=0.35 -> 8, x=0.40 -> 7, x=0.50 -> 5.
    """

    linea: tuple = ((0.35, 0.0), (0.35, 1.0))


@dataclass
class FilaCascadeConfig:
    """Punto 4: deteccion de filas sobre una zona de espera.

    La ROI se declara POR CLIP en corpus/rois.json y no aca: cada camara tiene su
    geometria. Un clip sin ROI declarada no se evalua para filas.
    """

    activo: bool = os.environ.get("AMTA_FILAS", "1") not in ("0", "false", "no")
    rois_path: Path = CORPUS_DIR / "rois.json"


@dataclass
class CascadeConfig:
    """Config unica del cascade. Parametrizable por env vars AMTA_*."""

    corpus_dir: Path = CORPUS_DIR
    raw_dir: Path = RAW_DIR
    results_dir: Path = RESULTS_DIR
    models_dir: Path = MODELS_DIR
    costs_db: Path = RESULTS_DIR / "costs.db"
    rtsp_url: str = os.environ.get("AMTA_RTSP_URL", "rtsp://localhost:8554/cam1")
    # Tarifa CPU. Por defecto 0.0 A PROPOSITO: cualquier cost_usd que se cite como
    # numero real exige fijar esta variable a una tarifa cloud verificada primero.
    cpu_usd_per_hour: float = float(os.environ.get("AMTA_CPU_USD_PER_HOUR", "0.0"))
    # Idem para la GPU: el Nivel 2 corre ahi, asi que sin esta tarifa el costo en
    # dolares del Nivel 2 es 0 por construccion aunque los ms sean reales.
    gpu_usd_per_hour: float = float(os.environ.get("AMTA_GPU_USD_PER_HOUR", "0.0"))
    # OpenCV multi-hilo contamina la medicion de CPU time: 1 hilo = numero reproducible.
    cv_num_threads: int = int(os.environ.get("AMTA_CV_THREADS", "1"))
    motion: MotionConfig = field(default_factory=MotionConfig)
    person: PersonConfig = field(default_factory=PersonConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    conteo: ConteoConfig = field(default_factory=ConteoConfig)
    reid: ReidCascadeConfig = field(default_factory=ReidCascadeConfig)
    filas: FilaCascadeConfig = field(default_factory=FilaCascadeConfig)


def load_cascade_config() -> CascadeConfig:
    """Punto de entrada unico para obtener la config del cascade."""
    return CascadeConfig()
