"""Tests del reporte. Lo critico: que NO publique un numero cuando el corpus
no es representativo, y que si lo publique cuando lo es.
"""

from __future__ import annotations

from core.cascade.config import CascadeConfig
from core.cascade.catalog.schema import (
    CameraHeight,
    ClipMeta,
    CrowdDensity,
    Lighting,
    LocationType,
)
from core.cascade.reporte import MIN_CLIPS_COMERCIALES, construir_reporte
from core.cascade.stage1_motion.runner import ClipMotionStats


def _clip(clip_id: str, tipo: LocationType) -> ClipMeta:
    return ClipMeta(
        clip_id=clip_id,
        filename=f"{clip_id}.mp4",
        source="test",
        camera_height=CameraHeight.CEILING,
        lighting=Lighting.BRIGHT,
        crowd_density=CrowdDensity.SPARSE,
        location_type=tipo,
        duration_s=10.0,
        width=640,
        height=480,
        fps=25.0,
        codec="h264",
        n_frames=250,
        sha256="deadbeef",
    )


def _stats(clip_id: str, kept: int, discarded: int) -> ClipMotionStats:
    s = ClipMotionStats(clip_id=clip_id)
    s.total_frames = kept + discarded
    s.frames_kept = kept
    s.frames_discarded = discarded
    s.pct_motion = round(100.0 * kept / (kept + discarded), 2)
    s.cpu_time_ms = 100.0
    return s


def _resumen():
    return {"motion_detection": {"n_eventos": 1, "cpu_total_ms": 100.0,
                                 "cpu_medio_ms": 100.0, "wall_total_ms": 110.0,
                                 "cost_usd_total": 0.0}}


def test_corpus_solo_interino_no_publica_numero():
    clips = [_clip("a", LocationType.OTHER), _clip("b", LocationType.OTHER)]
    stats = [_stats("a", 90, 10), _stats("b", 80, 20)]
    txt = construir_reporte(stats, clips, _resumen(), CascadeConfig())

    assert "PENDIENTE — corpus no representativo" in txt
    assert "Hay 0 clips de interior comercial" in txt
    # El 85% agregado de los clips interinos NO puede aparecer como resultado.
    assert "% de los frames** de un interior comercial" not in txt


def test_corpus_suficiente_publica_el_numero():
    clips = [_clip(f"c{i}", LocationType.STORE) for i in range(MIN_CLIPS_COMERCIALES)]
    # 25 de 100 frames con movimiento en cada clip -> 25.00 %
    stats = [_stats(f"c{i}", 25, 75) for i in range(MIN_CLIPS_COMERCIALES)]
    txt = construir_reporte(stats, clips, _resumen(), CascadeConfig())

    assert "PENDIENTE" not in txt
    assert "**25.00 % de los frames**" in txt
    assert f"n={MIN_CLIPS_COMERCIALES} clips" in txt


def test_los_clips_other_no_contaminan_el_agregado():
    """Un clip interino con 100% movimiento no debe mover el numero publicado."""
    clips = [_clip(f"c{i}", LocationType.STORE) for i in range(MIN_CLIPS_COMERCIALES)]
    stats = [_stats(f"c{i}", 25, 75) for i in range(MIN_CLIPS_COMERCIALES)]
    clips.append(_clip("interino", LocationType.OTHER))
    stats.append(_stats("interino", 100, 0))  # 100% movimiento, camara en mano

    txt = construir_reporte(stats, clips, _resumen(), CascadeConfig())
    assert "**25.00 % de los frames**" in txt  # sigue siendo 25, no sube


def test_protocolo_y_aviso_de_tarifa_van_siempre():
    clips = [_clip("a", LocationType.OTHER)]
    stats = [_stats("a", 1, 1)]
    txt = construir_reporte(stats, clips, _resumen(), CascadeConfig())

    # El umbral de flicker tiene que ir escrito para que nadie cite el resultado sin el.
    assert "flicker_fg_ratio=0.5" in txt
    assert "cv_threads=1" in txt
    # Y el aviso de que el dolar es cero por construccion.
    assert "cost_usd es 0 por construccion" in txt
