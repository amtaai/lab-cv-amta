"""Tests de la medicion de iluminacion.

Se testea la REGLA sobre metricas ya medidas, no la lectura de video: las
metricas crudas quedan en ClipMeta.meta, asi que la regla se puede revisar
despues sin releer los clips.
"""

from __future__ import annotations

import numpy as np

from core.cascade.catalog.medir import (
    FRAC_EXTREMO_MIN,
    LUMA_DIM_MAX,
    clasificar_iluminacion,
    metricas_de_frames,
)
from core.cascade.catalog.schema import Lighting


def test_escena_oscura_es_dim():
    assert clasificar_iluminacion(40.0, 0.0, 30.0) is Lighting.DIM


def test_escena_clara_sin_sombras_duras_es_bright():
    # Muchos highlights quemados pero casi nada aplastado: es brillante, no mixta.
    assert clasificar_iluminacion(182.0, 4.0, 0.15) is Lighting.BRIGHT


def test_quemado_y_aplastado_a_la_vez_es_mixed():
    assert clasificar_iluminacion(118.5, 0.5, 5.7) is Lighting.MIXED


def test_dim_gana_sobre_mixed_si_la_escena_es_oscura():
    """Escena oscura con ambos extremos sigue siendo DIM: el problema dominante
    para el detector es la falta de luz."""
    assert clasificar_iluminacion(LUMA_DIM_MAX - 1, 5.0, 5.0) is Lighting.DIM


def test_hacen_falta_LOS_DOS_extremos_para_mixed():
    justo = FRAC_EXTREMO_MIN
    assert clasificar_iluminacion(150.0, justo, justo) is Lighting.MIXED
    # Solo uno de los dos no alcanza.
    assert clasificar_iluminacion(150.0, justo, justo - 0.1) is Lighting.BRIGHT
    assert clasificar_iluminacion(150.0, justo - 0.1, justo) is Lighting.BRIGHT


def test_metricas_de_frames_sobre_frames_sinteticos():
    negro = np.zeros((48, 64, 3), dtype=np.uint8)
    m = metricas_de_frames([negro])
    assert m["luma_media"] == 0.0
    assert m["frac_oscuro_pct"] == 100.0
    assert m["frac_quemado_pct"] == 0.0

    blanco = np.full((48, 64, 3), 255, dtype=np.uint8)
    m = metricas_de_frames([blanco])
    assert m["luma_media"] == 255.0
    assert m["frac_quemado_pct"] == 100.0

    # Medio y medio: los dos extremos al 50%.
    mitad = np.zeros((48, 64, 3), dtype=np.uint8)
    mitad[:, 32:] = 255
    m = metricas_de_frames([mitad])
    assert 120.0 < m["luma_media"] < 135.0
    assert m["frac_quemado_pct"] == 50.0
    assert m["frac_oscuro_pct"] == 50.0
    assert clasificar_iluminacion(**{k: m[k] for k in
                                     ("luma_media", "frac_quemado_pct", "frac_oscuro_pct")}
                                  ) is Lighting.MIXED


def test_metricas_sin_frames_no_revienta():
    m = metricas_de_frames([])
    assert m["luma_media"] == 0.0
    assert m["n_frames_muestreados"] == 0


def test_gris_medio_uniforme_es_bright():
    """Sin extremos en ninguno de los dos lados: brillante y parejo."""
    gris = np.full((48, 64, 3), 128, dtype=np.uint8)
    m = metricas_de_frames([gris])
    assert m["frac_quemado_pct"] == 0.0
    assert m["frac_oscuro_pct"] == 0.0
    assert clasificar_iluminacion(m["luma_media"], 0.0, 0.0) is Lighting.BRIGHT
