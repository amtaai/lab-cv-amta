"""Fine-tuning y destilacion del modelo target. CORRE EN COLAB, no en local.

A partir del dataset auto-etiquetado (pseudo-labels), entrena/afina un detector
target rapido y lo destila a un checkpoint compacto que se baja a `models/`
para correr la inferencia local de la web demo.

IMPORTANTE: este modulo asume GPU grande (Colab 80GB). No invocar en runtime
local (ver core.config.Runtime).
"""

from __future__ import annotations

from pathlib import Path

from core.config import Config


class Trainer:
    """Orquesta el fine-tune + destilacion en Colab. TODO: bucle de entrenamiento."""

    def __init__(self, cfg: Config):
        if cfg.is_local:
            raise RuntimeError("El entrenamiento solo corre en Colab, no en runtime local")
        self.cfg = cfg

    def fine_tune(self, dataset_dir: Path) -> Path:
        """Afina el detector target sobre el dataset de pseudo-labels."""
        raise NotImplementedError("Pendiente: bucle de fine-tuning del target")

    def distill(self, teacher_ckpt: Path) -> Path:
        """Destila el teacher a un modelo compacto para inferencia local."""
        raise NotImplementedError("Pendiente: destilacion -> distilled_detector.pt")
