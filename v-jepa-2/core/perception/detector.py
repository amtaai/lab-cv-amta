"""Detector open-vocabulary: arranca la deteccion desde prompts en lenguaje natural.

Permite que la camara empiece con una "idea basica" expresada en texto
("una persona", "alguien con un carrito") sin entrenamiento previo de clases.

Backends candidatos (a implementar):
- Grounding DINO  (IDEA-Research, arxiv 2303.05499)
- YOLO-World      (arxiv 2401.17270) -- mas rapido, mejor para edge

Corre en el lado pesado (Colab) durante el bootstrapping; el modelo destilado
final lo reemplaza para la inferencia local.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.types import Detection


class OpenVocabDetector(ABC):
    """Interfaz comun para detectores guiados por prompts de texto."""

    @abstractmethod
    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Detecta en `frame` (HxWx3 RGB) los conceptos descritos en `prompts`."""
        raise NotImplementedError


class GroundingDinoDetector(OpenVocabDetector):
    """Adapter para Grounding DINO. TODO: cargar pesos y mapear salida a Detection."""

    def __init__(self, weights: str | None = None, box_threshold: float = 0.3):
        self.weights = weights
        self.box_threshold = box_threshold

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        raise NotImplementedError("Pendiente: integrar Grounding DINO")


class YoloWorldDetector(OpenVocabDetector):
    """Adapter para YOLO-World (tier edge / baja latencia). TODO."""

    def __init__(self, weights: str | None = None, conf: float = 0.3):
        self.weights = weights
        self.conf = conf

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        raise NotImplementedError("Pendiente: integrar YOLO-World")
