"""Tests de la deteccion de filas (punto 4).

El conteo dentro de la region es de `ultralytics.solutions.QueueManager`. Lo que
se testea aca es lo que ultralytics NO hace: distinguir una fila de un pasillo.
Sin eso, cualquiera que cruce la zona cuenta como si estuviera esperando.
"""

from __future__ import annotations

import numpy as np

from core.perception.filas import DetectorFila, FilaConfig, cargar_rois
from core.types import Detection

ANCHO, ALTO = 640, 480
# ROI propia del test, no la del proyecto: las de corpus/rois.json describen
# camaras concretas y van a cambiar con el corpus.
ROI = ((0.30, 0.30), (0.98, 0.30), (0.98, 0.98), (0.30, 0.98))


def _frame():
    return np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)


def _det(x, y=200, track_id=1, w=60, h=140):
    d = Detection(bbox=(float(x), float(y), float(x + w), float(y + h)),
                  label="person", score=0.9, track_id=track_id)
    d.meta["confirmado"] = True
    return d


def _fila(dwell_min_s=0.4, min_personas=2, fps=25.0, quieto_frac=0.06):
    return DetectorFila(ROI, ANCHO, ALTO, fps=fps,
                        cfg=FilaConfig(dwell_min_s=dwell_min_s,
                                       quieto_frac=quieto_frac,
                                       min_personas=min_personas))


def test_gente_parada_en_la_zona_es_una_fila():
    f = _fila()
    for _ in range(20):
        e = f.actualizar(_frame(), [_det(300, track_id=1), _det(400, track_id=2)])
    assert e.en_zona == 2
    assert e.en_fila == 2
    assert e.hay_fila is True
    assert f.hubo_fila is True
    assert f.max_en_fila == 2


def test_gente_que_solo_pasa_no_es_una_fila():
    """La diferencia que ultralytics no hace: estar adentro no es esperar."""
    f = _fila()
    ultimo = None
    for x in range(200, 620, 20):  # cruzan la zona caminando
        ultimo = f.actualizar(_frame(), [_det(x, track_id=1), _det(x, y=300, track_id=2)])
    assert ultimo.en_fila == 0
    assert f.hubo_fila is False


def test_una_sola_persona_parada_no_es_una_fila():
    """Es alguien siendo atendido, no una cola. Es el caso del clip de caja."""
    f = _fila(min_personas=2)
    for _ in range(20):
        e = f.actualizar(_frame(), [_det(300, track_id=1)])
    assert e.en_fila == 1     # esta esperando...
    assert e.hay_fila is False  # ...pero sola no hace fila
    assert f.hubo_fila is False


def test_fuera_de_la_roi_no_cuenta_aunque_este_quieto():
    f = _fila(min_personas=1)
    for _ in range(20):  # x=20 cae a la izquierda de la ROI (empieza en x=192)
        e = f.actualizar(_frame(), [_det(20, track_id=1)])
    assert (e.en_zona, e.en_fila) == (0, 0)


def test_salir_de_la_roi_reinicia_la_permanencia():
    """Si vuelve, vuelve a esperar: no arrastra el tiempo de la visita anterior."""
    f = _fila(dwell_min_s=0.4, min_personas=1)  # 10 frames
    for _ in range(9):
        f.actualizar(_frame(), [_det(300, track_id=1)])
    for _ in range(3):
        f.actualizar(_frame(), [_det(20, track_id=1)])  # se va de la ROI
    assert f.actualizar(_frame(), [_det(300, track_id=1)]).en_fila == 0


def test_el_umbral_de_permanencia_sale_de_los_fps():
    """2 s no son la misma cantidad de frames a 25 que a 30 fps."""
    assert _fila(dwell_min_s=2.0, fps=25.0).dwell_min_frames == 50
    assert _fila(dwell_min_s=2.0, fps=30.0).dwell_min_frames == 60


def test_el_tamano_de_la_fila_es_el_maximo_simultaneo():
    f = _fila()
    for _ in range(15):
        f.actualizar(_frame(), [_det(300, track_id=1), _det(400, track_id=2)])
    for _ in range(15):
        f.actualizar(_frame(), [_det(300, track_id=1), _det(400, track_id=2),
                                _det(500, track_id=3)])
    assert f.max_en_fila == 3


def test_las_detecciones_sin_track_id_no_cuentan():
    f = _fila(min_personas=1)
    suelta = Detection(bbox=(300.0, 200.0, 360.0, 340.0), label="person", score=0.5)
    for _ in range(20):
        e = f.actualizar(_frame(), [suelta])
    assert (e.en_zona, e.en_fila) == (0, 0)


def test_un_frame_sin_detecciones_no_rompe():
    f = _fila()
    e = f.actualizar(_frame(), [])
    assert (e.en_zona, e.en_fila, e.hay_fila) == (0, 0, False)


# ---- las ROI declaradas del proyecto ---------------------------------------

def test_las_rois_del_corpus_son_normalizadas(tmp_path):
    """En pixeles, la misma ROI cubre otra cosa en cada resolucion."""
    from core.cascade.config import load_cascade_config

    rois = cargar_rois(load_cascade_config().filas.rois_path)
    assert rois, "corpus/rois.json deberia declarar al menos una ROI"
    for clip, poly in rois.items():
        for x, y in poly:
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, f"{clip}: {(x, y)}"


def test_un_clip_sin_roi_declarada_no_se_evalua(tmp_path):
    """Mejor no dar un numero que dar uno sobre un rectangulo arbitrario."""
    rois = cargar_rois(tmp_path / "no_existe.json")
    assert rois == {}
