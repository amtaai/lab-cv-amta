"""Tests del reporte.

Lo critico ya no es negarse a publicar el numero, sino que el numero SIEMPRE
salga acompanado de su alcance: de que corpus sale y para que no sirve.
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
    s.pct_motion = round(100.0 * kept / (kept + discarded), 2) if kept + discarded else 0.0
    s.cpu_time_ms = 100.0
    return s


def _resumen():
    return {"motion_detection": {"n_eventos": 1, "cpu_total_ms": 100.0,
                                 "cpu_medio_ms": 100.0, "wall_total_ms": 110.0,
                                 "cost_usd_total": 0.0}}


def test_corpus_no_comercial_publica_el_numero_pero_con_alcance():
    clips = [_clip("a", LocationType.OTHER), _clip("b", LocationType.OTHER)]
    stats = [_stats("a", 90, 10), _stats("b", 80, 20)]
    txt = construir_reporte(stats, clips, _resumen(), CascadeConfig())

    # El numero se publica (85 % sobre 200 frames utiles)...
    assert "85.00 % de los frames del corpus" in txt
    # ...pero nunca sin la advertencia de alcance.
    assert "**ALCANCE.**" in txt
    assert "NO son interiores comerciales" in txt
    assert "no debe citarse como tal" in txt


def test_corpus_comercial_suficiente_no_lleva_advertencia():
    clips = [_clip(f"c{i}", LocationType.STORE) for i in range(MIN_CLIPS_COMERCIALES)]
    stats = [_stats(f"c{i}", 25, 75) for i in range(MIN_CLIPS_COMERCIALES)]
    txt = construir_reporte(stats, clips, _resumen(), CascadeConfig())

    assert "25.00 % de los frames del corpus" in txt
    assert "**ALCANCE.**" not in txt


def test_estimacion_por_ciclo_separa_vacios_de_activos():
    """Los clips casi sin movimiento son el regimen 'escena vacia'."""
    clips = [_clip(f"v{i}", LocationType.OTHER) for i in range(3)]
    clips += [_clip(f"a{i}", LocationType.OTHER) for i in range(3)]
    # 3 clips vacios (2 % de movimiento) y 3 activos (80 %).
    stats = [_stats(f"v{i}", 2, 98) for i in range(3)]
    stats += [_stats(f"a{i}", 80, 20) for i in range(3)]
    txt = construir_reporte(stats, clips, _resumen(), CascadeConfig())

    assert "## Estimacion por ciclo de actividad" in txt
    assert "escena vacia : n=3" in txt
    assert "escena activa: n=3" in txt
    assert "p_v = 0.0200" in txt
    assert "p_a = 0.8000" in txt
    # La tabla de ciclos tiene que traer las cinco filas.
    for f in ("5 %", "10 %", "15 %", "25 %", "50 %"):
        assert f"| {f} |" in txt


def test_ciclo_bajo_da_mas_descarte_que_el_corpus_crudo():
    """Con poca actividad el descarte sube: es el punto de toda la estimacion."""
    clips = [_clip("v", LocationType.OTHER), _clip("a", LocationType.OTHER)]
    stats = [_stats("v", 0, 100), _stats("a", 90, 10)]
    txt = construir_reporte(stats, clips, _resumen(), CascadeConfig())

    fila_5 = [l for l in txt.splitlines() if l.startswith("| 5 %")][0]
    fila_50 = [l for l in txt.splitlines() if l.startswith("| 50 %")][0]
    desc_5 = float(fila_5.split("|")[3].strip().replace(" %", ""))
    desc_50 = float(fila_50.split("|")[3].strip().replace(" %", ""))
    assert desc_5 > desc_50  # menos actividad -> mas descarte


def test_protocolo_y_aviso_de_tarifa_van_siempre():
    clips = [_clip("a", LocationType.OTHER)]
    stats = [_stats("a", 1, 1)]
    txt = construir_reporte(stats, clips, _resumen(), CascadeConfig())

    # El umbral de flicker tiene que ir escrito para que nadie cite el resultado sin el.
    assert "flicker_fg_ratio=0.5" in txt
    assert "cv_threads=1" in txt
    assert "min_area_frac=0.0016" in txt
    # Y el aviso de que el dolar es cero por construccion.
    assert "cost_usd es 0 por construccion" in txt
