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
class CascadeConfig:
    """Config unica del cascade. Parametrizable por env vars AMTA_*."""

    corpus_dir: Path = CORPUS_DIR
    raw_dir: Path = RAW_DIR
    results_dir: Path = RESULTS_DIR
    costs_db: Path = RESULTS_DIR / "costs.db"
    rtsp_url: str = os.environ.get("AMTA_RTSP_URL", "rtsp://localhost:8554/cam1")
    # Tarifa CPU. Por defecto 0.0 A PROPOSITO: cualquier cost_usd que se cite como
    # numero real exige fijar esta variable a una tarifa cloud verificada primero.
    cpu_usd_per_hour: float = float(os.environ.get("AMTA_CPU_USD_PER_HOUR", "0.0"))
    # OpenCV multi-hilo contamina la medicion de CPU time: 1 hilo = numero reproducible.
    cv_num_threads: int = int(os.environ.get("AMTA_CV_THREADS", "1"))
    motion: MotionConfig = field(default_factory=MotionConfig)


def load_cascade_config() -> CascadeConfig:
    """Punto de entrada unico para obtener la config del cascade."""
    return CascadeConfig()
