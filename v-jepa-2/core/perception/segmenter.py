"""Segmentacion en video con SAM 2 (arxiv 2408.00714).

SAM2 aporta mascaras promptables CON memoria temporal: se le da un prompt
(caja/punto) en un frame y propaga la mascara por el resto del clip. Es la
pieza clave para seguir personas frame a frame en el video del mall.

Soporta los tres "tipos" de segmentacion del proyecto cuando se combina con
el detector y el tracker:
- semantica  (clase por pixel)
- instancia  (una mascara por persona)  <- la principal para contar/seguir
- panoptica  (instancia + semantica de fondo)
"""

from __future__ import annotations

import numpy as np

from core.types import Detection


class Sam2Segmenter:
    """Adapter sobre SAM2VideoPredictor. TODO: cargar checkpoint y memoria."""

    def __init__(self, checkpoint: str | None = None, model_cfg: str | None = None):
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self._state = None  # estado de memoria temporal del predictor

    def init_video(self, first_frame: np.ndarray) -> None:
        """Inicializa el estado de memoria sobre el primer frame del clip."""
        raise NotImplementedError("Pendiente: SAM2 init_state")

    def add_prompts(self, detections: list[Detection]) -> list[Detection]:
        """Refina cada deteccion con una mascara a partir de su bbox (prompt)."""
        raise NotImplementedError("Pendiente: SAM2 add_new_points_or_box")

    def propagate(self, frame: np.ndarray) -> list[Detection]:
        """Propaga las mascaras existentes al siguiente frame usando memoria temporal."""
        raise NotImplementedError("Pendiente: SAM2 propagate_in_video")
