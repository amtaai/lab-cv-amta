"""Configuracion central del detector intuitivo.

Define donde corre cada cosa (local vs Colab), que tier de modelos se usa,
rutas de checkpoints y los umbrales que gobiernan el lazo de auto-correccion.

Topologia (ver CLAUDE.md):
- LOCAL  -> web demo + inferencia del modelo destilado. Sin entrenamiento.
- COLAB  -> heavy lifting: V-JEPA 2, SAM2, VLM, pseudo-labeling y fine-tuning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"  # checkpoints destilados bajados de Colab
DATA_DIR = ROOT / "data"  # clips de muestra + store de pseudo-labels


class Runtime(str, Enum):
    """Donde se esta ejecutando el codigo."""

    LOCAL = "local"  # maquina del usuario: solo inferencia + demo
    COLAB = "colab"  # GPU 80GB: entrenamiento y lazo completo


class Tier(str, Enum):
    """Tier de modelos. Define tamano vs latencia/precision."""

    SERVER = "server"  # Colab/GPU grande: ViT-g/G, SAM2-large, InternVL
    EDGE = "edge"  # liviano: ViT-L, SAM2-tiny, Florence-2, YOLO-World


# Catalogo de backbones V-JEPA 2 / 2.1 disponibles por PyTorch Hub.
# (ver third_party/vjepa2/hubconf.py). 2.1 trae dense features -> mejor
# para localizacion/segmentacion que 2.0.
VJEPA_HUB_ENTRYPOINTS = {
    Tier.SERVER: "vjepa2_1_vit_giant_384",
    Tier.EDGE: "vjepa2_1_vit_large_384",
}


@dataclass
class LoopThresholds:
    """Umbrales que gobiernan el lazo de auto-correccion (anti confirmation-bias)."""

    detector_min_score: float = 0.30  # descartar detecciones debajo de esto
    pseudo_label_min_score: float = 0.70  # solo pseudo-etiquetar lo muy confiable
    vlm_verify_min_agreement: float = 0.60  # acuerdo minimo del VLM para aceptar
    max_vlm_iterations: int = 3  # mas iteraciones => sobre-correccion
    human_review_fraction: float = 0.05  # % de muestras a revision humana


@dataclass
class Config:
    runtime: Runtime = Runtime(os.environ.get("AMTA_RUNTIME", "local"))
    tier: Tier = Tier(os.environ.get("AMTA_TIER", "server"))
    thresholds: LoopThresholds = field(default_factory=LoopThresholds)
    distilled_model_path: Path = MODELS_DIR / "distilled_detector.pt"

    @property
    def is_local(self) -> bool:
        return self.runtime is Runtime.LOCAL

    @property
    def vjepa_entrypoint(self) -> str:
        return VJEPA_HUB_ENTRYPOINTS[self.tier]


def load_config() -> Config:
    """Punto de entrada unico para obtener la config (parametrizable por env vars)."""
    return Config()
