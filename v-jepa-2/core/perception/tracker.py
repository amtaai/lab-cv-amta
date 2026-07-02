"""Tracking multi-objeto para asociar detecciones entre frames y contar instancias.

Asigna un track_id estable a cada persona para poder contar, medir dwell-time
y aplicar reglas por zona. Backend candidato: ByteTrack (rapido) o FairMOT.
"""

from __future__ import annotations

from core.types import Detection, Track


class Tracker:
    """Adapter de tracking (ByteTrack por defecto). TODO: integrar libreria."""

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Asocia `detections` a tracks existentes y devuelve las detecciones con track_id."""
        raise NotImplementedError("Pendiente: integrar ByteTrack/FairMOT")

    @property
    def unique_count(self) -> int:
        """Cantidad de instancias unicas vistas hasta ahora."""
        return len(self._tracks)
