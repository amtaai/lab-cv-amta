"""Medidor de costo reutilizable por cualquier etapa del cascade.

Se instrumenta una etapa con una linea:
    with tracker.track("motion_detection", clip_id=cid, n_frames=n):
        ...

cpu_time_ms usa time.process_time_ns(): tiempo de CPU, no de reloj, para que
esperar el stream RTSP no se facture como computo. OJO: process_time es del
PROCESO entero, asi que solo es valido en etapas secuenciales — si en el futuro
se paraleliza una etapa hay que pasar a time.thread_time_ns() por worker.

Los eventos se bufferean y se escriben con executemany: a nivel de frame, el
INSERT costaria mas que el propio MOG2 y contaminaria la medicion.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from core.cascade.cost.schema import COLUMNAS_NUEVAS, DDL, DDL_INDICES, CostEvent

MS_POR_HORA = 3_600_000.0


class CostTracker:
    """Acumula CostEvent y los persiste en SQLite.

    La db es un log ACUMULATIVO entre corridas. Para que un reporte describa solo
    la corrida actual, cada tracker genera un run_id y resumen_por_stage() filtra
    por el. Sin eso, correr dos veces duplicaba los agregados del reporte.
    """

    def __init__(self, db_path: Path, cpu_usd_per_hour: float = 0.0,
                 gpu_usd_per_hour: float = 0.0,
                 flush_every: int = 200, run_id: str | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cpu_usd_per_hour = cpu_usd_per_hour
        # El Nivel 2 corre en GPU: sin esta tarifa su costo en dolares es 0
        # aunque los milisegundos sean reales.
        self.gpu_usd_per_hour = gpu_usd_per_hour
        self.flush_every = flush_every
        self.run_id = run_id or str(uuid.uuid4())
        self._buffer: list[CostEvent] = []
        self._con = sqlite3.connect(self.db_path)
        self._con.executescript(DDL)
        self._migrar()
        self._con.executescript(DDL_INDICES)
        self._con.commit()

    def _migrar(self) -> None:
        """Agrega las columnas nuevas a una db creada por una version anterior."""
        existentes = {r[1] for r in self._con.execute("PRAGMA table_info(cost_events)")}
        for col, tipo in COLUMNAS_NUEVAS.items():
            if col not in existentes:
                self._con.execute(f"ALTER TABLE cost_events ADD COLUMN {col} {tipo}")

    def _costo(self, ev: CostEvent) -> float:
        """USD del evento: CPU y GPU se cobran por separado y se suman."""
        return (ev.cpu_time_ms / MS_POR_HORA * self.cpu_usd_per_hour
                + ev.gpu_time_ms / MS_POR_HORA * self.gpu_usd_per_hour)

    @contextmanager
    def track(self, stage: str, **meta):
        """Mide un bloque y encola el evento. Persiste aunque el bloque falle."""
        ev = CostEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            run_id=self.run_id,
            meta=dict(meta),
        )
        t_cpu0 = time.process_time_ns()
        t_wall0 = time.perf_counter_ns()
        try:
            yield ev
        finally:
            ev.cpu_time_ms = (time.process_time_ns() - t_cpu0) / 1e6
            ev.wall_time_ms = (time.perf_counter_ns() - t_wall0) / 1e6
            ev.cost_usd += self._costo(ev)
            self._buffer.append(ev)
            if len(self._buffer) >= self.flush_every:
                self.flush()

    def registrar(self, stage: str, cpu_time_ms: float = 0.0, gpu_time_ms: float = 0.0,
                  tokens_used: int = 0, wall_time_ms: float = 0.0, **meta) -> CostEvent:
        """Encola un evento YA medido por fuera. Devuelve el evento encolado.

        track() envuelve un bloque, y eso no sirve en dos casos que ya aparecen:
        cuando dos etapas corren INTERCALADAS dentro del mismo loop (el Nivel 2
        se ejecuta salteado entre frames del Nivel 1, asi que sus bloques se
        solapan y anidar track() haria que cada uno midiera el tiempo del otro),
        y cuando el tiempo lo reporta otro reloj (eventos CUDA, latencia de una
        API). En los dos casos se acumula afuera y se registra el total aca.

        El cost_usd se calcula con la misma tarifa que en track().
        """
        ev = CostEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            run_id=self.run_id,
            cpu_time_ms=cpu_time_ms,
            gpu_time_ms=gpu_time_ms,
            tokens_used=tokens_used,
            wall_time_ms=wall_time_ms,
            meta=dict(meta),
        )
        ev.cost_usd += self._costo(ev)
        self._buffer.append(ev)
        if len(self._buffer) >= self.flush_every:
            self.flush()
        return ev

    def tracked(self, stage: str):
        """Decorador equivalente a track(), para instrumentar una funcion entera."""

        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*a, **kw):
                with self.track(stage, func=fn.__name__):
                    return fn(*a, **kw)

            return wrapper

        return deco

    def flush(self) -> int:
        """Vuelca el buffer a SQLite. Devuelve cuantos eventos escribio."""
        if not self._buffer:
            return 0
        filas = [
            (e.event_id, e.run_id, e.timestamp, e.stage, e.cpu_time_ms, e.gpu_time_ms,
             e.tokens_used, e.cost_usd, e.wall_time_ms,
             json.dumps(e.meta, ensure_ascii=False))
            for e in self._buffer
        ]
        self._con.executemany(
            "INSERT OR REPLACE INTO cost_events (event_id, run_id, timestamp, stage,"
            " cpu_time_ms, gpu_time_ms, tokens_used, cost_usd, wall_time_ms, meta)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            filas,
        )
        self._con.commit()
        n = len(self._buffer)
        self._buffer.clear()
        return n

    def resumen_por_stage(self, solo_esta_corrida: bool = True) -> dict:
        """Agregados por etapa: n eventos, cpu total y medio, costo total.

        Por defecto solo la corrida actual: la db acumula entre corridas y un
        reporte que sumara todo describiria una historia, no la medicion de hoy.
        """
        self.flush()
        sql = (
            "SELECT stage, COUNT(*), SUM(cpu_time_ms), AVG(cpu_time_ms),"
            " SUM(wall_time_ms), SUM(cost_usd), SUM(gpu_time_ms), AVG(gpu_time_ms)"
            " FROM cost_events"
        )
        params: tuple = ()
        if solo_esta_corrida:
            sql += " WHERE run_id = ?"
            params = (self.run_id,)
        cur = self._con.execute(sql + " GROUP BY stage", params)
        return {
            r[0]: {
                "n_eventos": r[1],
                "cpu_total_ms": round(r[2] or 0.0, 3),
                "cpu_medio_ms": round(r[3] or 0.0, 4),
                "wall_total_ms": round(r[4] or 0.0, 3),
                "cost_usd_total": round(r[5] or 0.0, 8),
                "gpu_total_ms": round(r[6] or 0.0, 3),
                "gpu_medio_ms": round(r[7] or 0.0, 4),
            }
            for r in cur.fetchall()
        }

    def close(self) -> None:
        """Vuelca lo pendiente y cierra la conexion."""
        self.flush()
        self._con.close()
