"""Corre el Nivel 1 sobre un clip o sobre el stream RTSP, instrumentado con costo.

El evento de costo es POR CLIP, no por frame: a nivel de frame el INSERT en
SQLite costaria mas que el propio MOG2. n_frames va en meta, asi que el costo
por frame se deriva igual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from core.cascade.config import CascadeConfig
from core.cascade.cost.tracker import CostTracker
from core.cascade.stage1_motion.detector import MotionDetector, MotionResult


@dataclass
class ClipMotionStats:
    """Stats de un clip: cuantos frames se descartaron y cuanto costo hacerlo."""

    clip_id: str
    total_frames: int = 0
    warmup_frames: int = 0
    frames_kept: int = 0  # con movimiento relevante
    frames_discarded: int = 0  # sin movimiento (excluye warmup)
    flicker_suspect_frames: int = 0
    pct_discarded: float = 0.0
    pct_motion: float = 0.0
    cpu_time_ms: float = 0.0
    meta: dict = field(default_factory=dict)


def _acumular(stats: ClipMotionStats, r: MotionResult) -> None:
    stats.total_frames += 1
    if r.is_warmup:
        stats.warmup_frames += 1
        return
    if r.flicker_suspect:
        stats.flicker_suspect_frames += 1
    if r.has_motion:
        stats.frames_kept += 1
    else:
        stats.frames_discarded += 1


def _cerrar(stats: ClipMotionStats) -> ClipMotionStats:
    utiles = stats.frames_kept + stats.frames_discarded
    if utiles:
        stats.pct_discarded = round(100.0 * stats.frames_discarded / utiles, 2)
        stats.pct_motion = round(100.0 * stats.frames_kept / utiles, 2)
    return stats


def procesar_clip(path: Path, clip_id: str, cfg: CascadeConfig,
                  tracker: CostTracker) -> ClipMotionStats:
    """Procesa un archivo de video entero. Un evento de costo por clip."""
    det = MotionDetector(cfg.motion)
    stats = ClipMotionStats(clip_id=clip_id)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        stats.meta["error"] = f"no se pudo abrir {path}"
        return stats
    try:
        with tracker.track("motion_detection", clip_id=clip_id, fuente="file") as ev:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _acumular(stats, det.process(frame))
            ev.meta["n_frames"] = stats.total_frames
    finally:
        cap.release()
    stats.cpu_time_ms = round(ev.cpu_time_ms, 3)
    return _cerrar(stats)


def procesar_stream(url: str, cfg: CascadeConfig, tracker: CostTracker,
                    max_frames: int = 300) -> ClipMotionStats:
    """Procesa max_frames del stream RTSP. Valida que el simulador es equivalente."""
    det = MotionDetector(cfg.motion)
    stats = ClipMotionStats(clip_id="rtsp_stream")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        stats.meta["error"] = f"no se pudo abrir el stream {url}"
        return stats
    try:
        with tracker.track("motion_detection", clip_id="rtsp_stream", fuente="rtsp") as ev:
            while stats.total_frames < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                _acumular(stats, det.process(frame))
            ev.meta["n_frames"] = stats.total_frames
    finally:
        cap.release()
    stats.cpu_time_ms = round(ev.cpu_time_ms, 3)
    return _cerrar(stats)
