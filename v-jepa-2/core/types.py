"""Tipos de datos compartidos por todo el pipeline de percepcion y el lazo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Bounding box en formato xyxy absoluto (pixeles).
BBox = tuple[float, float, float, float]


@dataclass
class Detection:
    """Una deteccion en un frame: caja, clase propuesta y confianza."""

    bbox: BBox
    label: str
    score: float
    mask: Optional[np.ndarray] = None  # mascara binaria HxW (la pone el segmenter)
    track_id: Optional[int] = None  # lo asigna el tracker
    verified: Optional[bool] = None  # lo decide el VLM verificador
    meta: dict = field(default_factory=dict)


@dataclass
class Track:
    """Trayectoria temporal de una instancia a lo largo de varios frames."""

    track_id: int
    label: str
    history: list[Detection] = field(default_factory=list)


@dataclass
class FrameResult:
    """Salida del orquestador para un frame: detecciones + metadatos."""

    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
