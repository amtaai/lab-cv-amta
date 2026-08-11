"""Nivel 1 del cascade: deteccion de movimiento (el filtro mas barato)."""

from __future__ import annotations

from core.cascade.stage1_motion.detector import MotionDetector, MotionResult

__all__ = ["MotionDetector", "MotionResult"]
