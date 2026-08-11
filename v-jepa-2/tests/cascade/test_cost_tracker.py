"""Tests del medidor de costo. Lo importante: que cpu_time_ms sea REAL y no cero."""

from __future__ import annotations

import sqlite3

from core.cascade.cost.tracker import CostTracker


def _quemar_cpu(n: int = 2_000_000) -> int:
    """Trabajo real de CPU para que process_time avance de forma medible."""
    s = 0
    for i in range(n):
        s += i
    return s


def test_track_produce_cpu_time_no_cero(tmp_path):
    tr = CostTracker(tmp_path / "costs.db")
    with tr.track("motion_detection", n_frames=10) as ev:
        _quemar_cpu()
    tr.close()
    assert ev.cpu_time_ms > 0.0
    assert ev.wall_time_ms > 0.0
    assert ev.stage == "motion_detection"
    assert ev.meta["n_frames"] == 10


def test_evento_persiste_con_el_esquema_del_spec(tmp_path):
    db = tmp_path / "costs.db"
    tr = CostTracker(db)
    with tr.track("motion_detection"):
        pass
    tr.close()

    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(cost_events)")}
    assert {"event_id", "timestamp", "stage", "cpu_time_ms",
            "gpu_time_ms", "tokens_used", "cost_usd"} <= cols
    assert con.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0] == 1
    con.close()


def test_cost_usd_escala_con_la_tarifa(tmp_path):
    tr = CostTracker(tmp_path / "c.db", cpu_usd_per_hour=3600.0)  # 1 usd por segundo cpu
    with tr.track("x") as ev:
        _quemar_cpu()
    tr.close()
    esperado = ev.cpu_time_ms / 3_600_000.0 * 3600.0
    assert abs(ev.cost_usd - esperado) < 1e-9
    assert ev.cost_usd > 0.0


def test_cost_usd_es_cero_si_no_hay_tarifa(tmp_path):
    tr = CostTracker(tmp_path / "c.db")  # cpu_usd_per_hour=0.0 por defecto
    with tr.track("x") as ev:
        _quemar_cpu()
    tr.close()
    assert ev.cpu_time_ms > 0.0  # el tiempo si es real
    assert ev.cost_usd == 0.0  # el dolar es cero por construccion


def test_decorador_registra_evento(tmp_path):
    tr = CostTracker(tmp_path / "c.db")

    @tr.tracked("etapa_decorada")
    def trabajo(n):
        return sum(range(n))

    trabajo(100_000)
    tr.close()
    con = sqlite3.connect(tmp_path / "c.db")
    filas = con.execute("SELECT stage FROM cost_events").fetchall()
    con.close()
    assert filas == [("etapa_decorada",)]


def test_excepcion_igual_persiste_el_evento(tmp_path):
    tr = CostTracker(tmp_path / "c.db")
    try:
        with tr.track("falla"):
            raise ValueError("boom")
    except ValueError:
        pass
    tr.close()
    con = sqlite3.connect(tmp_path / "c.db")
    n = con.execute("SELECT COUNT(*) FROM cost_events WHERE stage='falla'").fetchone()[0]
    con.close()
    assert n == 1


def test_buffer_se_vuelca_al_llegar_a_flush_every(tmp_path):
    db = tmp_path / "c.db"
    tr = CostTracker(db, flush_every=3)
    for _ in range(3):
        with tr.track("bulk"):
            pass
    # Ya deberia haber volcado sin llamar a close()
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0] == 3
    con.close()
    tr.close()


def test_resumen_por_stage_agrega(tmp_path):
    tr = CostTracker(tmp_path / "c.db")
    for _ in range(2):
        with tr.track("a"):
            _quemar_cpu(500_000)
    with tr.track("b"):
        _quemar_cpu(500_000)
    resumen = tr.resumen_por_stage()
    tr.close()
    assert resumen["a"]["n_eventos"] == 2
    assert resumen["b"]["n_eventos"] == 1
    assert resumen["a"]["cpu_total_ms"] > 0.0
