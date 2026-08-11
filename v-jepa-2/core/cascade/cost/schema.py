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
    timestamp    TEXT NOT NULL,                 -- ISO-8601 UTC
    stage        TEXT NOT NULL,
    cpu_time_ms  REAL NOT NULL DEFAULT 0.0,
    gpu_time_ms  REAL NOT NULL DEFAULT 0.0,
    tokens_used  INTEGER NOT NULL DEFAULT 0,
    cost_usd     REAL NOT NULL DEFAULT 0.0,
    wall_time_ms REAL NOT NULL DEFAULT 0.0,
    meta         TEXT NOT NULL DEFAULT '{}'     -- JSON libre (n_frames, clip_id, ...)
);
CREATE INDEX IF NOT EXISTS idx_cost_events_stage ON cost_events(stage);
"""


@dataclass
class CostEvent:
    """Un evento de costo. Los tiempos los llena el CostTracker al salir del bloque."""

    event_id: str
    timestamp: str
    stage: str
    cpu_time_ms: float = 0.0
    gpu_time_ms: float = 0.0  # lo llenan las etapas GPU (semana 2+)
    tokens_used: int = 0  # lo llenan las llamadas a VLM (semana 3+)
    cost_usd: float = 0.0
    wall_time_ms: float = 0.0  # reloj de pared: incluye I/O y espera de red
    meta: dict = field(default_factory=dict)
