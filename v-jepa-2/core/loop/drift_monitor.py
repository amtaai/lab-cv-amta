"""Guardrails del lazo: detecta drift de pseudo-labels y dispara revision humana.

El riesgo central de un lazo auto-supervisado es el confirmation bias: el
modelo refuerza sus propios errores. Este monitor sigue estadisticas de
confianza/acuerdo en el tiempo y marca cuando el lazo se esta degradando,
ademas de samplear un % de detecciones para revision humana.
"""

from __future__ import annotations

from collections import deque

from core.config import LoopThresholds
from core.types import Detection


class DriftMonitor:
    """Vigila la salud del lazo de auto-correccion."""

    def __init__(self, thresholds: LoopThresholds, window: int = 500):
        self.thresholds = thresholds
        self._scores: deque[float] = deque(maxlen=window)

    def observe(self, detections: list[Detection]) -> None:
        """Registra las confianzas del batch actual."""
        self._scores.extend(d.score for d in detections)

    def is_drifting(self) -> bool:
        """True si la confianza media cae por debajo del umbral de pseudo-label."""
        raise NotImplementedError("Pendiente: estadistico de drift + alerta")

    def sample_for_human_review(self, detections: list[Detection]) -> list[Detection]:
        """Devuelve un subconjunto a revisar manualmente (human-in-the-loop)."""
        raise NotImplementedError("Pendiente: muestreo segun human_review_fraction")
