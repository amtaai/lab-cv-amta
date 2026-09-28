"""Nivel 1 del cascade: deteccion de movimiento con MOG2.

Es el filtro mas barato: descarta los frames sin nada interesante antes de que
lleguen a las etapas caras (deteccion de objetos, VLM). El numero que produce
—que fraccion de frames tiene movimiento relevante— es el multiplicador que
despues gobierna el costo mensual por camara.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.cascade.config import MotionConfig


@dataclass
class MotionResult:
    """Veredicto de un frame."""

    frame_index: int
    has_motion: bool
    fg_ratio: float  # fraccion de pixeles marcados como foreground [0,1]
    largest_area_px: int  # area del contorno mas grande
    flicker_suspect: bool = False  # cambio de luz global, no movimiento
    is_warmup: bool = False  # MOG2 todavia aprendiendo: excluir de las stats
    meta: dict = field(default_factory=dict)


class MotionDetector:
    """Detector de movimiento MOG2 con guarda de flicker y warmup explicito."""

    def __init__(self, cfg: MotionConfig | None = None) -> None:
        self.cfg = cfg or MotionConfig()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.reset()

    def reset(self) -> None:
        """Reinicia el modelo de fondo y el contador de frames."""
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history,
            varThreshold=self.cfg.var_threshold,
            detectShadows=self.cfg.detect_shadows,
        )
        self._i = -1

    def process(self, frame: np.ndarray) -> MotionResult:
        """Clasifica un frame BGR como movimiento relevante o descarte."""
        self._i += 1
        mask = self._bg.apply(frame)
        # MOG2 marca las sombras con 127 cuando detectShadows=True: fuera.
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        fg_ratio = float(np.count_nonzero(mask)) / mask.size
        es_warmup = self._i < self.cfg.warmup_frames

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_max = int(max((cv2.contourArea(c) for c in contornos), default=0))

        # Umbral de area RELATIVO al frame: un umbral absoluto en pixeles hace
        # que el mismo valor sea mucho mas sensible a alta resolucion.
        umbral_area = self.cfg.min_area_frac * mask.size

        # Guarda de flicker: si medio frame se enciende de golpe es la luz, no
        # una persona. Es la falsa positiva dominante en interiores comerciales.
        flicker = (not es_warmup) and fg_ratio > self.cfg.flicker_fg_ratio

        hay_movimiento = (
            not es_warmup
            and not flicker
            and area_max >= umbral_area
            and fg_ratio >= self.cfg.motion_fg_ratio_min
        )

        return MotionResult(
            frame_index=self._i,
            has_motion=hay_movimiento,
            fg_ratio=round(fg_ratio, 6),
            largest_area_px=area_max,
            flicker_suspect=flicker,
            is_warmup=es_warmup,
            meta={"umbral_area_px": round(umbral_area, 1)},
        )
