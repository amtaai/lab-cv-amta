"""Orquestador: une percepcion -> razonamiento -> lazo de auto-correccion.

Flujo por frame (lado pesado / Colab):
    1. detector open-vocab propone cajas desde prompts en lenguaje natural
    2. SAM2 refina/propaga mascaras con memoria temporal
    3. tracker asigna track_id estables (contar / seguir)
    4. (opcional) encoder V-JEPA2 aporta features semanticas / de novedad
    5. VLM verifica y corrige las detecciones dudosas
    6. pseudo_labeler emite labels confiables; drift_monitor vigila la salud
Periodicamente el trainer (en Colab) reentrena y destila un modelo compacto.

En runtime LOCAL el orquestador corre en modo inferencia: usa unicamente el
modelo destilado ya entrenado (sin detector pesado, sin VLM, sin lazo).
"""

from __future__ import annotations

import numpy as np

from core.config import Config, load_config
from core.types import FrameResult


class IntuitiveDetector:
    """Punto de entrada de alto nivel del sistema. TODO: cablear componentes."""

    def __init__(self, cfg: Config | None = None, prompts: list[str] | None = None):
        self.cfg = cfg or load_config()
        self.prompts = prompts or ["una persona"]
        # Los componentes se instancian perezosamente segun runtime (local vs colab).

    def process_frame(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        """Procesa un frame y devuelve las detecciones resultantes."""
        if self.cfg.is_local:
            return self._process_local(frame, frame_index)
        return self._process_full_loop(frame, frame_index)

    def _process_local(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        """Inferencia liviana con el modelo destilado (web demo)."""
        raise NotImplementedError("Pendiente: cargar models/distilled_detector.pt e inferir")

    def _process_full_loop(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        """Pipeline completo con auto-correccion (Colab)."""
        raise NotImplementedError("Pendiente: cablear detector->sam2->tracker->vlm->loop")
