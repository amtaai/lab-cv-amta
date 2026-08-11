"""Tests del Nivel 1. Los dos que importan: warmup excluido y flicker no cuenta."""

from __future__ import annotations

import numpy as np

from core.cascade.config import MotionConfig
from core.cascade.stage1_motion.detector import MotionDetector


def _correr(det, frames):
    return [det.process(f) for f in frames]


def test_escena_estatica_no_reporta_movimiento(frame_negro):
    cfg = MotionConfig(warmup_frames=10)
    det = MotionDetector(cfg)
    res = _correr(det, [frame_negro.copy() for _ in range(60)])
    utiles = [r for r in res if not r.is_warmup]
    assert len(utiles) == 50
    assert sum(r.has_motion for r in utiles) == 0


def test_objeto_en_movimiento_reporta_movimiento(frame_negro, frame_con_rectangulo):
    cfg = MotionConfig(warmup_frames=10, min_area_px=200)
    det = MotionDetector(cfg)
    _correr(det, [frame_negro.copy() for _ in range(30)])  # aprende el fondo
    res = _correr(det, [frame_con_rectangulo(x, 200) for x in range(50, 400, 25)])
    assert sum(r.has_motion for r in res) >= 5


def test_flicker_global_no_cuenta_como_movimiento(frame_negro):
    cfg = MotionConfig(warmup_frames=10, flicker_fg_ratio=0.5)
    det = MotionDetector(cfg)
    _correr(det, [frame_negro.copy() for _ in range(30)])
    # subida de brillo uniforme en todo el frame: es luz, no movimiento
    brillante = np.full((480, 640, 3), 200, dtype=np.uint8)
    res = _correr(det, [brillante.copy() for _ in range(3)])
    assert res[0].flicker_suspect is True
    assert res[0].has_motion is False
    assert res[0].fg_ratio > 0.5


def test_objeto_diminuto_se_descarta_por_area(frame_negro):
    cfg = MotionConfig(warmup_frames=10, min_area_px=5000)
    det = MotionDetector(cfg)
    _correr(det, [frame_negro.copy() for _ in range(30)])
    chico = frame_negro.copy()
    chico[200:210, 200:210] = 255  # 100 px, muy debajo de min_area_px
    r = det.process(chico)
    assert r.has_motion is False


def test_reset_vuelve_a_warmup(frame_negro):
    det = MotionDetector(MotionConfig(warmup_frames=5))
    _correr(det, [frame_negro.copy() for _ in range(20)])
    det.reset()
    assert det.process(frame_negro.copy()).is_warmup is True


def test_warmup_se_marca_en_los_primeros_n_frames(frame_negro):
    det = MotionDetector(MotionConfig(warmup_frames=7))
    res = _correr(det, [frame_negro.copy() for _ in range(15)])
    assert [r.is_warmup for r in res[:7]] == [True] * 7
    assert [r.is_warmup for r in res[7:]] == [False] * 8
    # Durante el warmup nunca se reporta movimiento, pase lo que pase.
    assert sum(r.has_motion for r in res[:7]) == 0
