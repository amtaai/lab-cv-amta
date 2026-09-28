"""Tests de las metricas de evaluacion (punto 5).

Precision, recall y F1 son faciles de calcular mal en los bordes —sin
detecciones, sin ground truth, dos predicciones sobre la misma persona— y esos
son justo los casos que aparecen en un corpus donde muchos frames estan vacios.
"""

from __future__ import annotations

import pytest

from core.cascade.eval.metricas import Conteos, barrer_umbrales, evaluar_frame, iou

CAJA = (0.2, 0.2, 0.4, 0.8)


def test_iou_de_cajas_identicas_y_disjuntas():
    assert iou(CAJA, CAJA) == 1.0
    assert iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_una_prediccion_que_matchea_es_un_acierto():
    c = evaluar_frame([(CAJA, 0.9)], [(CAJA, False)])
    assert (c.tp, c.fp, c.fn) == (1, 0, 0)
    assert c.precision == 1.0 and c.recall == 1.0 and c.f1 == 1.0


def test_una_prediccion_donde_no_hay_nadie_es_falso_positivo():
    c = evaluar_frame([(CAJA, 0.9)], [])
    assert (c.tp, c.fp, c.fn) == (0, 1, 0)
    assert c.precision == 0.0


def test_una_persona_no_detectada_es_falso_negativo():
    c = evaluar_frame([], [(CAJA, False)])
    assert (c.tp, c.fp, c.fn) == (0, 0, 1)
    assert c.recall == 0.0


def test_un_frame_vacio_sin_predicciones_no_rompe():
    """La mayoria de los frames de una camara real son asi."""
    c = evaluar_frame([], [])
    assert (c.tp, c.fp, c.fn) == (0, 0, 0)
    assert c.precision == 0.0 and c.recall == 0.0 and c.f1 == 0.0


def test_dos_predicciones_sobre_la_misma_persona_dan_un_falso_positivo():
    """Una caja real se empareja una sola vez: la duplicada no puede ser acierto."""
    c = evaluar_frame([(CAJA, 0.9), ((0.21, 0.21, 0.41, 0.81), 0.8)], [(CAJA, False)])
    assert (c.tp, c.fp, c.fn) == (1, 1, 0)


def test_la_caja_de_mayor_confianza_se_queda_con_el_acierto():
    """Se ordena por score: si no, el acierto se lo lleva una prediccion peor."""
    buena = (0.2, 0.2, 0.4, 0.8)
    floja = (0.25, 0.25, 0.45, 0.85)
    c = evaluar_frame([(floja, 0.4), (buena, 0.95)], [(buena, False)])
    assert c.tp == 1
    # La que matcheo es la de 0.95; subir el umbral por encima de 0.4 no cambia nada.
    assert evaluar_frame([(floja, 0.4), (buena, 0.95)], [(buena, False)],
                         conf_min=0.5).tp == 1


def test_un_solapamiento_pobre_no_cuenta_como_acierto():
    apenas = (0.35, 0.2, 0.55, 0.8)  # solapa poco con CAJA
    assert iou(apenas, CAJA) < 0.5
    c = evaluar_frame([(apenas, 0.9)], [(CAJA, False)])
    assert (c.tp, c.fp, c.fn) == (0, 1, 1)


def test_el_umbral_de_confianza_filtra_predicciones():
    c = evaluar_frame([(CAJA, 0.3)], [(CAJA, False)], conf_min=0.5)
    assert (c.tp, c.fn) == (0, 1)


def test_una_caja_dificil_ignorada_no_suma_ni_resta():
    """Si el anotador no estaba seguro, el detector no deberia pagar por eso."""
    c = evaluar_frame([(CAJA, 0.9)], [(CAJA, True)], ignorar_dificiles=True)
    assert (c.tp, c.fp, c.fn) == (0, 0, 0)
    assert c.ignoradas == 1


def test_contando_las_dificiles_si_cuentan():
    c = evaluar_frame([(CAJA, 0.9)], [(CAJA, True)], ignorar_dificiles=False)
    assert (c.tp, c.fp, c.fn) == (1, 0, 0)


def test_no_detectar_una_dificil_ignorada_no_es_falso_negativo():
    c = evaluar_frame([], [(CAJA, True)], ignorar_dificiles=True)
    assert c.fn == 0
    assert evaluar_frame([], [(CAJA, True)], ignorar_dificiles=False).fn == 1


def test_los_conteos_se_suman_entre_frames():
    a = Conteos(tp=2, fp=1, fn=1)
    b = Conteos(tp=3, fp=0, fn=2)
    total = a + b
    assert (total.tp, total.fp, total.fn) == (5, 1, 3)
    assert total.precision == pytest.approx(5 / 6)
    assert total.recall == pytest.approx(5 / 8)


def test_f1_es_la_media_armonica():
    c = Conteos(tp=3, fp=1, fn=1)  # p = 0.75, r = 0.75
    assert c.f1 == pytest.approx(0.75)
    c2 = Conteos(tp=1, fp=1, fn=3)  # p = 0.5, r = 0.25
    assert c2.f1 == pytest.approx(2 * 0.5 * 0.25 / 0.75)


def test_subir_el_umbral_no_puede_subir_la_recall():
    """Propiedad de la curva: filtrar predicciones solo puede perder aciertos."""
    frames = [([(CAJA, 0.3), ((0.6, 0.2, 0.8, 0.8), 0.9)],
               [(CAJA, False), ((0.6, 0.2, 0.8, 0.8), False)])]
    res = barrer_umbrales(frames, [0.1, 0.5, 0.95])
    recalls = [res[u].recall for u in (0.1, 0.5, 0.95)]
    assert recalls == sorted(recalls, reverse=True)
