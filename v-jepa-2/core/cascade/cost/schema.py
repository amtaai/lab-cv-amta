"""Esquema de eventos de costo. Una fila por invocacion instrumentada.

Los campos gpu_time_ms / tokens_used / cost_usd quedan en cero esta semana
(Nivel 1 es solo CPU) pero existen desde ya: las etapas caras de las semanas
siguientes (deteccion, VLM) los llenan sin migrar la tabla.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DDL = """
CREATE TABLE IF NOT EXISTS cost_events (
    event_id     TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL DEFAULT '',      -- una corrida del pipeline
    timestamp    TEXT NOT NULL,                 -- ISO-8601 UTC
    stage        TEXT NOT NULL,
    cpu_time_ms  REAL NOT NULL DEFAULT 0.0,
    gpu_time_ms  REAL NOT NULL DEFAULT 0.0,
    tokens_used  INTEGER NOT NULL DEFAULT 0,
    cost_usd     REAL NOT NULL DEFAULT 0.0,
    wall_time_ms REAL NOT NULL DEFAULT 0.0,
    meta         TEXT NOT NULL DEFAULT '{}'     -- JSON libre (n_frames, clip_id, ...)
);
"""

# Los indices van DESPUES de la migracion: en una db vieja, indexar run_id antes
# de que la columna exista revienta con "no such column".
DDL_INDICES = """
CREATE INDEX IF NOT EXISTS idx_cost_events_stage ON cost_events(stage);
CREATE INDEX IF NOT EXISTS idx_cost_events_run ON cost_events(run_id);
"""

# Columnas agregadas despues de la v1 de la tabla. CREATE TABLE IF NOT EXISTS no
# las agrega a una db que ya existe, asi que el tracker las mete con ALTER TABLE.
COLUMNAS_NUEVAS = {
    "run_id": "TEXT NOT NULL DEFAULT ''",
}


@dataclass
class CostEvent:
    """Un evento de costo. Los tiempos los llena el CostTracker al salir del bloque."""

    event_id: str
    timestamp: str
    stage: str
    run_id: str = ""  # agrupa los eventos de una misma corrida
    cpu_time_ms: float = 0.0
    gpu_time_ms: float = 0.0  # lo llenan las etapas GPU (semana 2+)
    tokens_used: int = 0  # lo llenan las llamadas a VLM (semana 3+)
    cost_usd: float = 0.0
    wall_time_ms: float = 0.0  # reloj de pared: incluye I/O y espera de red
    meta: dict = field(default_factory=dict)
