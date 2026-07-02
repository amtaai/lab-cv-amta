"""Extractor de features de V-JEPA 2 / 2.1 (encoder congelado).

V-JEPA 2 es un world model de video auto-supervisado: NO produce cajas ni
mascaras, sino embeddings espaciotemporales. Aca lo usamos como backbone
congelado del que extraemos features densas (V-JEPA 2.1 mejora justo la
calidad de las dense features -> util para localizacion/segmentacion).

Usos en el lazo:
- features semanticas para una cabeza de refinamiento/decoder downstream
- firmas de apariencia para re-identificacion / similitud entre instancias
- senal de "novedad" para descubrir clases no vistas (open-world)

Se carga via PyTorch Hub (ver third_party/vjepa2/hubconf.py):
    torch.hub.load('facebookresearch/vjepa2', cfg.vjepa_entrypoint)
"""

from __future__ import annotations

import numpy as np


class VJepa2Encoder:
    """Wrapper del encoder congelado de V-JEPA 2/2.1. TODO: cargar via hub."""

    def __init__(self, entrypoint: str, device: str = "cuda"):
        self.entrypoint = entrypoint
        self.device = device
        self._model = None
        self._preprocessor = None

    def load(self) -> None:
        """Carga el backbone y el preprocessor desde PyTorch Hub (congelados, eval())."""
        raise NotImplementedError("Pendiente: torch.hub.load + freeze + eval")

    def extract_features(self, clip: np.ndarray) -> np.ndarray:
        """Devuelve los embeddings espaciotemporales para un clip (T,H,W,3)."""
        raise NotImplementedError("Pendiente: forward del encoder congelado")
