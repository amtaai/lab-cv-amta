"""Genera pseudo-labels a partir de detecciones ya verificadas por el VLM.

Solo se convierten en pseudo-labels las detecciones que pasan los umbrales
(score alto + verificadas), para no contaminar el dataset con errores que
luego el modelo aprenderia (confirmation bias).

Escribe al store de datos (data/) que despues consume el trainer en Colab.
"""

from __future__ import annotations

from pathlib import Path

from core.config import DATA_DIR, LoopThresholds
from core.types import Detection


class PseudoLabeler:
    """Filtra detecciones verificadas y las persiste como pseudo-labels."""

    def __init__(self, thresholds: LoopThresholds, store_dir: Path = DATA_DIR / "pseudo_labels"):
        self.thresholds = thresholds
        self.store_dir = store_dir

    def is_trustworthy(self, det: Detection) -> bool:
        """Una deteccion es pseudo-etiquetable si esta verificada y es confiable."""
        return bool(det.verified) and det.score >= self.thresholds.pseudo_label_min_score

    def emit(self, frame_index: int, detections: list[Detection]) -> int:
        """Persiste las detecciones confiables como pseudo-labels. Devuelve cuantas emitio."""
        raise NotImplementedError("Pendiente: serializar pseudo-labels (COCO/YOLO) al store")
