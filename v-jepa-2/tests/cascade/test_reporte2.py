"""Tests del reporte del Nivel 2.

Igual que en el Nivel 1: lo critico no es el numero sino que nunca salga sin su
alcance, y que la cuenta de la economia de la cascada este bien hecha.
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
from core.cascade.reporte2 import MIN_CLIPS_COMERCIALES, construir_reporte
from core.cascade.stage2_person.runner import ClipPersonStats


def _clip(clip_id: str, tipo: LocationType = LocationType.OTHER) -> ClipMeta:
    return ClipMeta(
        clip_id=clip_id, filename=f"{clip_id}.mp4", source="test",
        camera_height=CameraHeight.CEILING, lighting=Lighting.BRIGHT,
        crowd_density=CrowdDensity.SPARSE, location_type=tipo,
        duration_s=10.0, width=640, height=480, fps=25.0, codec="h264",
        n_frames=250, sha256="deadbeef",
    )


def _stats(clip_id, total=100, analizados=20, con_persona=15, det=30, personas=2,
           cpu_mov=700.0, gpu_per=240.0, cpu_reg=2.0):
    s = ClipPersonStats(clip_id=clip_id)
    s.total_frames = total
    s.frames_analizados = analizados
    s.frames_descartados = total - analizados
    s.frames_con_persona = con_persona
    s.n_detecciones = det
    s.n_personas_unicas = personas
    s.max_personas_simultaneas = 2
    s.conf_media = 0.8
    s.cpu_ms_motion = cpu_mov
    s.gpu_ms_person = gpu_per
    s.cpu_ms_tracking = cpu_reg
    return s


def _resumen():
    return {
        "motion_detection": {"n_eventos": 1, "cpu_total_ms": 700.0, "cpu_medio_ms": 700.0,
                             "wall_total_ms": 800.0, "cost_usd_total": 0.0,
                             "gpu_total_ms": 0.0, "gpu_medio_ms": 0.0},
        "person_detection": {"n_eventos": 1, "cpu_total_ms": 0.0, "cpu_medio_ms": 0.0,
                             "wall_total_ms": 240.0, "cost_usd_total": 0.0,
                             "gpu_total_ms": 240.0, "gpu_medio_ms": 240.0},
        "tracking": {"n_eventos": 1, "cpu_total_ms": 2.0, "cpu_medio_ms": 2.0,
                     "wall_total_ms": 2.0, "cost_usd_total": 0.0},
    }


def test_publica_deteccion_y_seguimiento():
    txt = construir_reporte([_stats("a")], [_clip("a")], _resumen(), CascadeConfig())
    assert "## Deteccion" in txt
    assert "## Seguimiento" in txt
    assert "personas unicas (tracks confirmados): **2**" in txt
    # El argumento de por que el seguimiento hace falta tiene que estar escrito.
    assert "se contarian como 30 personas" in txt
    # Y el conteo NUNCA sale sin decir cuanto se le puede creer.
    assert "no esta validado contra" in txt


def test_el_protocolo_va_siempre():
    """Nadie puede citar el numero sin saber con que modelo y umbrales salio."""
    txt = construir_reporte([_stats("a")], [_clip("a")], _resumen(), CascadeConfig())
    assert "protocolo: yolo11" in txt
    assert "imgsz=640" in txt
    assert "conf=0.35" in txt
    assert "bytetrack.yaml" in txt
    assert "min_hits=3" in txt


def test_dice_que_el_detector_es_el_del_repo():
    """Que quede escrito que no hay un modelo paralelo del cascade."""
    txt = construir_reporte([_stats("a")], [_clip("a")], _resumen(), CascadeConfig())
    assert "core/perception/" in txt
    assert "yolo_seg/" in txt and "yolo_world/" in txt


def test_corpus_no_comercial_lleva_alcance():
    txt = construir_reporte([_stats("a")], [_clip("a", LocationType.OTHER)],
                            _resumen(), CascadeConfig())
    assert "**ALCANCE.**" in txt
    assert "NO son interiores comerciales" in txt


def test_corpus_comercial_no_lleva_alcance():
    clips = [_clip(f"c{i}", LocationType.STORE) for i in range(MIN_CLIPS_COMERCIALES)]
    stats = [_stats(f"c{i}") for i in range(MIN_CLIPS_COMERCIALES)]
    txt = construir_reporte(stats, clips, _resumen(), CascadeConfig())
    assert "**ALCANCE.**" not in txt


def test_la_economia_de_la_cascada_esta_bien_calculada():
    """C1 se paga por todo frame (CPU); C2 solo por los que pasan (GPU)."""
    s = _stats("a", total=100, analizados=20, cpu_mov=700.0, gpu_per=240.0)
    txt = construir_reporte([s], [_clip("a")], _resumen(), CascadeConfig())

    # C1 = 700 ms cpu / 100 frames = 7.00 ; C2 = 240 ms gpu / 20 frames = 12.00
    assert "C1 (Nivel 1, CPU) = **7.00 ms/frame**" in txt
    assert "C2 (Nivel 2, GPU) = **12.00 ms/frame**" in txt
    # El ahorro de GPU es exactamente la fraccion de frames que el Nivel 1 tira.
    assert "**ahorro de GPU: 80.0 %**" in txt


def test_el_ahorro_de_gpu_es_la_fraccion_descartada():
    """Con las etapas en recursos distintos, el ahorro de GPU es 1 - p, sin mas."""
    s = _stats("a", total=100, analizados=25, cpu_mov=700.0, gpu_per=300.0)
    txt = construir_reporte([s], [_clip("a")], _resumen(), CascadeConfig())
    assert "**ahorro de GPU: 75.0 %**" in txt


def test_dice_cuantas_camaras_entran_por_gpu():
    """El numero que la semana 1 no podia dar, ahora en el recurso que limita."""
    txt = construir_reporte([_stats("a")], [_clip("a")], _resumen(), CascadeConfig())
    assert "## Cuantas camaras entran por GPU" in txt
    assert "camaras por GPU a 25 fps" in txt
    for f in ("5 %", "10 %", "15 %", "25 %", "50 %"):
        assert f"| {f} |" in txt


def test_menos_actividad_significa_mas_camaras_por_gpu():
    """Es el punto entero de la cascada: si no se cumple, algo esta mal."""
    vacio = _stats("v", total=100, analizados=1, con_persona=0, det=0, personas=0)
    activo = _stats("a", total=100, analizados=80, con_persona=70, det=100, personas=3)
    txt = construir_reporte([vacio, activo], [_clip("v"), _clip("a")],
                            _resumen(), CascadeConfig())

    fila_5 = [l for l in txt.splitlines() if l.startswith("| 5 %")][0]
    fila_50 = [l for l in txt.splitlines() if l.startswith("| 50 %")][0]
    cam_5 = float(fila_5.split("|")[5].strip())
    cam_50 = float(fila_50.split("|")[5].strip())
    assert cam_5 > cam_50  # menos actividad -> mas camaras por GPU
