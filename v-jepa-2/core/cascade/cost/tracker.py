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

from core.cascade.cost.schema import DDL, CostEvent

MS_POR_HORA = 3_600_000.0


class CostTracker:
    """Acumula CostEvent y los persiste en SQLite."""

    def __init__(self, db_path: Path, cpu_usd_per_hour: float = 0.0,
                 flush_every: int = 200) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cpu_usd_per_hour = cpu_usd_per_hour
        self.flush_every = flush_every
        self._buffer: list[CostEvent] = []
        self._con = sqlite3.connect(self.db_path)
        self._con.executescript(DDL)
        self._con.commit()

    @contextmanager
    def track(self, stage: str, **meta):
        """Mide un bloque y encola el evento. Persiste aunque el bloque falle."""
        ev = CostEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            meta=dict(meta),
        )
        t_cpu0 = time.process_time_ns()
        t_wall0 = time.perf_counter_ns()
        try:
            yield ev
        finally:
            ev.cpu_time_ms = (time.process_time_ns() - t_cpu0) / 1e6
            ev.wall_time_ms = (time.perf_counter_ns() - t_wall0) / 1e6
            ev.cost_usd += ev.cpu_time_ms / MS_POR_HORA * self.cpu_usd_per_hour
            self._buffer.append(ev)
            if len(self._buffer) >= self.flush_every:
                self.flush()

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
            (e.event_id, e.timestamp, e.stage, e.cpu_time_ms, e.gpu_time_ms,
             e.tokens_used, e.cost_usd, e.wall_time_ms,
             json.dumps(e.meta, ensure_ascii=False))
            for e in self._buffer
        ]
        self._con.executemany(
            "INSERT OR REPLACE INTO cost_events (event_id, timestamp, stage,"
            " cpu_time_ms, gpu_time_ms, tokens_used, cost_usd, wall_time_ms, meta)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            filas,
        )
        self._con.commit()
        n = len(self._buffer)
        self._buffer.clear()
        return n

    def resumen_por_stage(self) -> dict:
        """Agregados por etapa: n eventos, cpu total y medio, costo total."""
        self.flush()
        cur = self._con.execute(
            "SELECT stage, COUNT(*), SUM(cpu_time_ms), AVG(cpu_time_ms),"
            " SUM(wall_time_ms), SUM(cost_usd) FROM cost_events GROUP BY stage"
        )
        return {
            r[0]: {
                "n_eventos": r[1],
                "cpu_total_ms": round(r[2] or 0.0, 3),
                "cpu_medio_ms": round(r[3] or 0.0, 4),
                "wall_total_ms": round(r[4] or 0.0, 3),
                "cost_usd_total": round(r[5] or 0.0, 8),
            }
            for r in cur.fetchall()
        }

    def close(self) -> None:
        """Vuelca lo pendiente y cierra la conexion."""
        self.flush()
        self._con.close()
