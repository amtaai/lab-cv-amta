"""Tests de core/perception: el detector y el registro de tracks del proyecto.

Los pesos son pesados y necesitan GPU, asi que lo que se testea sin modelo es la
conversion de la salida de ultralytics a `core.types.Detection` y el registro de
tracks. Los tests que si necesitan el modelo se saltean solos.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.perception.detector import (
    BACKENDS,
    Yolo11Detector,
    YoloWorldDetector,
    crear_detector,
)
from core.perception.tracker import Tracker
from core.types import Detection


class _Cajas:
    """Imita ultralytics.engine.results.Boxes con lo minimo que se le lee."""

    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy = _Tensor(np.array(xyxy, dtype=np.float32))
        self.conf = _Tensor(np.array(conf, dtype=np.float32))
        self.cls = _Tensor(np.array(cls, dtype=np.float32))
        self.id = _Tensor(np.array(ids, dtype=np.float32)) if ids is not None else None

    def __len__(self):
        return len(self.xyxy.numpy())


class _Tensor:
    """Lo justo para que .cpu().numpy() funcione sin torch instalado."""

    def __init__(self, a):
        self._a = a

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Resultado:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


def _det(bbox=(0.0, 0.0, 10.0, 20.0), score=0.9, track_id=1, label="person"):
    return Detection(bbox=bbox, label=label, score=score, track_id=track_id)


# ---- conversion de la salida de ultralytics --------------------------------

def test_la_salida_de_ultralytics_se_convierte_a_detection():
    d = Yolo11Detector.__new__(Yolo11Detector)  # sin cargar pesos
    res = _Resultado(_Cajas([[10, 20, 30, 60]], [0.87], [0], [3]), {0: "person"})
    dets = d._a_detections(res)

    assert len(dets) == 1
    assert isinstance(dets[0], Detection)  # el tipo del proyecto, no uno nuevo
    assert dets[0].bbox == (10.0, 20.0, 30.0, 60.0)
    assert dets[0].label == "person"
    assert dets[0].score == pytest.approx(0.87)
    assert dets[0].track_id == 3


def test_sin_seguimiento_el_track_id_queda_en_none():
    """predict() no trae ids; solo track() los pone. No hay que inventarlos."""
    d = Yolo11Detector.__new__(Yolo11Detector)
    res = _Resultado(_Cajas([[0, 0, 5, 5]], [0.5], [0]), {0: "person"})
    assert d._a_detections(res)[0].track_id is None


def test_un_frame_sin_cajas_devuelve_lista_vacia():
    d = Yolo11Detector.__new__(Yolo11Detector)
    assert d._a_detections(_Resultado(_Cajas([], [], []), {0: "person"})) == []
    assert d._a_detections(_Resultado(None, {})) == []


def test_la_fabrica_conoce_los_dos_backends_del_repo():
    assert set(BACKENDS) == {"yolo11", "yoloworld"}
    assert isinstance(crear_detector("yolo11"), Yolo11Detector)
    assert isinstance(crear_detector("yoloworld"), YoloWorldDetector)
    with pytest.raises(ValueError):
        crear_detector("inexistente")


def test_los_pesos_y_umbrales_son_los_de_los_notebooks():
    """Si estos valores se despegan del notebook, los resultados no son comparables."""
    d = YoloWorldDetector()
    assert d.pesos.endswith("yolov8s-worldv2.pt")
    assert d.conf == 0.25
    assert Yolo11Detector().pesos.endswith("yolo11n-seg.pt")
    assert d.tracker == "bytetrack.yaml"


# ---- registro de tracks -----------------------------------------------------

def test_una_persona_en_muchos_frames_cuenta_una_vez():
    """El requisito del enunciado: no contar a la misma persona muchas veces."""
    reg = Tracker(min_hits=3)
    for _ in range(20):
        reg.update([_det(track_id=7)])
    assert reg.unique_count == 1


def test_dos_ids_distintos_cuentan_dos_personas():
    reg = Tracker(min_hits=1)
    for _ in range(5):
        reg.update([_det(track_id=1), _det(track_id=2)])
    assert reg.unique_count == 2


def test_min_hits_filtra_el_parpadeo_de_un_frame():
    reg = Tracker(min_hits=3)
    reg.update([_det(track_id=9)])
    assert reg.unique_count == 0
    reg.update([_det(track_id=9)])
    reg.update([_det(track_id=9)])
    assert reg.unique_count == 1


def test_una_persona_que_se_fue_sigue_contada():
    """El total es historico: si no, el conteo del clip baja cuando alguien sale."""
    reg = Tracker(min_hits=1, max_age=2)
    reg.update([_det(track_id=4)])
    for _ in range(10):
        reg.update([])
    assert reg.unique_count == 1
    assert reg.activos == []  # ya no esta en escena...
    assert reg.historia(4) is not None  # ...pero su trayectoria queda


def test_las_detecciones_sin_track_id_no_se_cuentan():
    """Vienen de detect() en vez de track(): contarlas duplicaria por frame."""
    reg = Tracker(min_hits=1)
    for _ in range(10):
        reg.update([_det(track_id=None)])
    assert reg.unique_count == 0


def test_la_deteccion_queda_marcada_como_confirmada():
    reg = Tracker(min_hits=2)
    assert reg.update([_det(track_id=1)])[0].meta["confirmado"] is False
    assert reg.update([_det(track_id=1)])[0].meta["confirmado"] is True


def test_reset_vacia_el_registro():
    reg = Tracker(min_hits=1)
    reg.update([_det(track_id=1)])
    reg.reset()
    assert reg.unique_count == 0
    assert reg.historia(1) is None


# ---- memoria del registro ---------------------------------------------------

def test_el_registro_no_guarda_las_mascaras():
    """Una mascara son ~960 KB; retenerlas por deteccion son GB y mata el proceso."""
    import numpy as np

    reg = Tracker(min_hits=1)
    d = _det(track_id=1)
    d.mask = np.zeros((384, 640), dtype=np.float32)
    reg.update([d])

    guardada = reg.historia(1).history[0]
    assert guardada.mask is None
    assert guardada.bbox == d.bbox  # el resto del dato si se conserva
    assert d.mask is not None       # y no se le rompe la mascara al que llamo


def test_la_historia_tiene_tope():
    """Sin tope, un stream en vivo acumula horas de detecciones y no para."""
    reg = Tracker(min_hits=1, max_historia=10)
    for _ in range(50):
        reg.update([_det(track_id=1)])
    assert len(reg.historia(1).history) == 10
    assert reg.unique_count == 1  # el conteo no se ve afectado por el tope
