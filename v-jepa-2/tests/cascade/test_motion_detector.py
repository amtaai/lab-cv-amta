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
    # 200 px sobre un frame de 640x480 = 0.065 % del area.
    cfg = MotionConfig(warmup_frames=10, min_area_frac=200 / (640 * 480))
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
    cfg = MotionConfig(warmup_frames=10, min_area_frac=5000 / (640 * 480))
    det = MotionDetector(cfg)
    _correr(det, [frame_negro.copy() for _ in range(30)])
    chico = frame_negro.copy()
    chico[200:210, 200:210] = 255  # 100 px, muy debajo del umbral
    r = det.process(chico)
    assert r.has_motion is False


def test_el_umbral_de_area_es_independiente_de_la_resolucion():
    """La MISMA escena a dos resoluciones tiene que dar el mismo veredicto.

    Regresion: con un umbral absoluto en pixeles (min_area_px=500), los clips de
    1920x1080 del corpus daban 99.9 % de movimiento contra 62.1 % de los de
    640x480 — no por su contenido, sino porque 500 px es 6.75x mas chico en
    proporcion a 1080p. El corpus mezcla resoluciones, asi que el umbral tiene
    que ser una fraccion del frame.
    """
    cfg = MotionConfig(warmup_frames=5, min_area_frac=0.0016)

    def correr(w, h, lado_frac):
        """Fondo negro y luego un cuadrado que ocupa lado_frac del ancho."""
        det = MotionDetector(cfg)
        fondo = np.zeros((h, w, 3), dtype=np.uint8)
        _correr(det, [fondo.copy() for _ in range(20)])
        lado = int(w * lado_frac)
        con_objeto = fondo.copy()
        con_objeto[h // 3 : h // 3 + lado, w // 3 : w // 3 + lado] = 255
        return det.process(con_objeto).has_motion

    # Un objeto que ocupa el 8 % del ancho: por encima del umbral en ambas.
    assert correr(640, 480, 0.08) is True
    assert correr(1920, 1080, 0.08) is True

    # Un objeto que ocupa el 1 % del ancho: por debajo del umbral en ambas.
    assert correr(640, 480, 0.01) is False
    assert correr(1920, 1080, 0.01) is False


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
