"""VLM verificador/critico: el corazon de la auto-correccion (VQA-in-the-loop).

Hallazgo clave del estado del arte: los VLM son malos auto-corrigiendose de
forma intrinseca, pero MUY buenos como verificadores externos. En vez de
confiar ciegamente en el detector, le preguntamos al VLM por cada deteccion
dudosa: "esto es realmente <label>?" y usamos su veredicto para aceptar,
descartar o corregir la etiqueta.

Backends candidatos:
- Florence-2 (Microsoft, ligero, grounding nativo) -> tier edge
- InternVL    (open-weights, multitarea)           -> tier server

Guardrails (ver core/config.LoopThresholds):
- cortar a max_vlm_iterations (mas iteraciones => sobre-correccion)
- aceptar solo con acuerdo >= vlm_verify_min_agreement
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import Detection


@dataclass
class Verdict:
    """Resultado de verificar una deteccion contra el VLM."""

    accepted: bool
    corrected_label: str | None
    agreement: float  # 0..1, confianza del VLM en el veredicto
    rationale: str = ""


class VlmVerifier:
    """Verificador VQA sobre recortes de detecciones. TODO: cargar VLM."""

    def __init__(self, backend: str = "florence-2", max_iterations: int = 3):
        self.backend = backend
        self.max_iterations = max_iterations
        self._model = None

    def verify(self, crop: np.ndarray, claimed_label: str) -> Verdict:
        """Pregunta al VLM si `crop` corresponde a `claimed_label`."""
        raise NotImplementedError("Pendiente: prompt VQA + parseo de respuesta")

    def verify_batch(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Verifica una lista de detecciones y marca `verified` / corrige `label`."""
        raise NotImplementedError("Pendiente: recortar bboxes y llamar verify()")
