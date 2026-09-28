"""Tests del conteo de personas por cruce de linea (punto 3).

La logica de cruce es de `ultralytics.solutions`; lo que se testea aca es que las
reglas queden como el enunciado las pide: cuando entra, cuando sale y cuando se
contabiliza.
"""

from __future__ import annotations

import numpy as np

from core.cascade.config import ConteoConfig
from core.perception.conteo import ContadorPersonas, a_pixeles
from core.types import Detection

ANCHO, ALTO = 640, 480

# Geometria PROPIA de los tests, no la de ConteoConfig: el default del proyecto
# esta puesto donde este corpus tiene trafico y va a cambiar con el corpus. Un
# test que dependa de el falla por un cambio de configuracion, no por un bug.
LINEA = ((0.5, 0.0), (0.5, 1.0))  # vertical en x=320


def _frame():
    return np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)


def _det(x, y=200.0, track_id=1, w=60.0, h=140.0, confirmado=True):
    d = Detection(bbox=(float(x), y, float(x) + w, y + h), label="person",
                  score=0.9, track_id=track_id)
    d.meta["confirmado"] = confirmado
    return d


def _recorrer(objs, xs, y=200.0, track_id=1):
    """Mueve una persona por las x dadas, alimentando a todos los objetos."""
    for x in xs:
        d = _det(x, y, track_id)
        for o in objs:
            o.actualizar(_frame(), [d])


# ---- geometria --------------------------------------------------------------

def test_el_default_del_proyecto_es_normalizado():
    """Si alguien lo pone en pixeles, la linea cae en otro lado en cada resolucion."""
    for punto in ConteoConfig().linea:
        assert all(0.0 <= v <= 1.0 for v in punto), punto


def test_la_geometria_es_relativa_a_la_resolucion():
    """En pixeles, la misma linea caeria en otro lado en 480p y en 1080p."""
    geom = ((0.5, 0.5), (1.0, 0.5))
    assert a_pixeles(geom, 640, 480)[0] == (320, 240)
    assert a_pixeles(geom, 1920, 1080)[0] == (960, 540)


# ---- punto 3: entra / sale / se contabiliza ---------------------------------

def test_cruzar_hacia_la_derecha_cuenta_como_entrada():
    c = ContadorPersonas(LINEA, ANCHO, ALTO)  # linea vertical en x=320
    _recorrer([c], range(200, 460, 20))
    assert c.entradas == 1
    assert c.salidas == 0


def test_cruzar_hacia_la_izquierda_cuenta_como_salida():
    c = ContadorPersonas(LINEA, ANCHO, ALTO)
    _recorrer([c], range(420, 160, -20))
    assert c.salidas == 1
    assert c.entradas == 0


def test_quien_no_cruza_no_se_cuenta():
    """Moverse dentro del cuadro no es entrar: hay que cruzar la linea."""
    c = ContadorPersonas(LINEA, ANCHO, ALTO)
    _recorrer([c], range(60, 200, 20))  # todo a la izquierda de x=320
    assert c.entradas == 0 and c.salidas == 0
    assert c.contados == 0


def test_una_persona_se_contabiliza_una_sola_vez():
    """Ir y venir sobre la linea no puede inflar el conteo del mismo track."""
    c = ContadorPersonas(LINEA, ANCHO, ALTO)
    _recorrer([c], list(range(200, 460, 20)) + list(range(440, 180, -20)))
    assert c.contados == 1
    assert c.entradas + c.salidas == 1


def test_dos_personas_distintas_se_cuentan_las_dos():
    c = ContadorPersonas(LINEA, ANCHO, ALTO)
    _recorrer([c], range(200, 460, 20), track_id=1)
    _recorrer([c], range(200, 460, 20), y=100.0, track_id=2)
    assert c.contados == 2
    assert c.entradas == 2


def test_los_tracks_sin_confirmar_no_se_cuentan():
    """Un parpadeo de un frame no es una persona que entro al local."""
    c = ContadorPersonas(LINEA, ANCHO, ALTO)
    for x in range(200, 460, 20):
        c.actualizar(_frame(), [_det(x, confirmado=False)])
    assert c.entradas == 0 and c.contados == 0


# ---- robustez ---------------------------------------------------------------

def test_una_deteccion_sin_track_id_no_rompe_nada():
    """ByteTrack no siempre asigna id: la caja llega con track_id None.

    Sin filtrarla, el conteo reventaba con TypeError al hacer int(None).
    """
    from core.perception.conteo import utiles

    suelta = Detection(bbox=(300.0, 200.0, 360.0, 340.0), label="person", score=0.5)
    assert suelta.track_id is None
    assert utiles([suelta]) == []

    c = ContadorPersonas(LINEA, ANCHO, ALTO)
    c.actualizar(_frame(), [suelta])
    assert c.entradas == 0


def test_un_frame_sin_detecciones_no_rompe_nada():
    c = ContadorPersonas(LINEA, ANCHO, ALTO)
    c.actualizar(_frame(), [])
    assert c.entradas == 0
