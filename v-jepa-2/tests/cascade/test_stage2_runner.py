"""Tests del cascade completo (Nivel 1 -> Nivel 2 -> registro de tracks).

El detector se inyecta como doble: lo que se testea aca es el CABLEADO —que el
Nivel 2 solo vea los frames que pasaron el filtro, que el costo de cada etapa
quede separado y en su recurso— no la calidad de YOLO ni de ByteTrack, que son
de ultralytics y ya vienen probados aguas arriba.
"""

from __future__ import annotations

import sqlite3

import cv2
import numpy as np
import pytest

from core.cascade.config import (
    CascadeConfig,
    MotionConfig,
    ReidCascadeConfig,
    TrackConfig,
)
from core.cascade.cost.tracker import CostTracker
from core.cascade.stage2_person.runner import procesar_clip, procesar_stream
from core.types import Detection


class DetectorFalso:
    """Imita OpenVocabDetector: cuenta llamadas y devuelve cajas con track_id."""

    def __init__(self, n_personas: int = 1, gpu_ms: float = 12.0):
        self.llamadas = 0
        self.n_personas = n_personas
        self.gpu_time_ms = 0.0
        self._gpu_ms = gpu_ms
        self.reseteado = 0
        self.prompts_vistos: list[str] | None = None

    def reset(self):
        self.reseteado += 1

    def track(self, frame, prompts):
        self.llamadas += 1
        self.prompts_vistos = list(prompts)
        self.gpu_time_ms = self._gpu_ms  # como si lo hubiera medido la GPU
        x = 50 + 4 * self.llamadas
        return [Detection(bbox=(x + 200.0 * i, 100.0, x + 60.0 + 200.0 * i, 240.0),
                          label="person", score=0.9, track_id=i + 1)
                for i in range(self.n_personas)]


def _video(tmp_path, n_frames=90, mover_desde=40):
    """Escribe un mp4: fondo fijo y, a partir de mover_desde, un rectangulo movil."""
    path = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 480))
    for i in range(n_frames):
        f = np.zeros((480, 640, 3), dtype=np.uint8)
        f[:] = 40
        if i >= mover_desde:
            x = 100 + (i - mover_desde) * 6
            f[150:330, x:x + 90] = 255
        vw.write(f)
    vw.release()
    return path


def _cfg(tmp_path, reid=False):
    """Config de test. El Nivel 3 va APAGADO salvo que el test lo pida.

    Encendido carga DINOv2 en GPU: son 22M de parametros y varios segundos por
    test, para probar un cableado que se verifica igual con un doble.
    """
    cfg = CascadeConfig()
    cfg.results_dir = tmp_path
    cfg.costs_db = tmp_path / "costs.db"
    cfg.motion = MotionConfig(warmup_frames=10)
    cfg.track = TrackConfig(min_hits=2)
    cfg.reid = ReidCascadeConfig(activo=reid)
    return cfg


def test_el_nivel2_solo_ve_los_frames_que_pasaron_el_filtro(tmp_path):
    """El punto entero de la cascada: el detector no puede correr sobre todo."""
    cfg = _cfg(tmp_path)
    tr = CostTracker(cfg.costs_db)
    fake = DetectorFalso()
    s = procesar_clip(_video(tmp_path), "c1", cfg, tr, detector=fake)
    tr.close()

    assert fake.llamadas == s.frames_analizados
    assert s.frames_descartados > 0  # hubo ahorro real
    assert s.frames_analizados + s.frames_descartados + s.warmup_frames == s.total_frames
    assert s.pct_analizado < 100.0


def test_se_le_pasan_los_prompts_de_la_config(tmp_path):
    """Con yoloworld el prompt es lo que decide que se detecta: no puede perderse."""
    cfg = _cfg(tmp_path)
    tr = CostTracker(cfg.costs_db)
    fake = DetectorFalso()
    procesar_clip(_video(tmp_path), "c1", cfg, tr, detector=fake)
    tr.close()
    assert fake.prompts_vistos == list(cfg.person.prompts)


def test_se_resetea_bytetrack_al_empezar_el_clip(tmp_path):
    """Sin esto la primera persona del clip hereda el ID de la ultima del anterior."""
    cfg = _cfg(tmp_path)
    tr = CostTracker(cfg.costs_db)
    fake = DetectorFalso()
    procesar_clip(_video(tmp_path), "c1", cfg, tr, detector=fake)
    tr.close()
    assert fake.reseteado == 1


def test_una_persona_en_todo_el_clip_se_cuenta_una_vez(tmp_path):
    """Requisito del enunciado: no contar a la misma persona muchas veces."""
    cfg = _cfg(tmp_path)
    tr = CostTracker(cfg.costs_db)
    s = procesar_clip(_video(tmp_path), "c1", cfg, tr, detector=DetectorFalso())
    tr.close()

    assert s.n_detecciones > s.n_personas_unicas  # muchas cajas...
    assert s.n_personas_unicas == 1  # ...pero una sola persona


def test_dos_personas_dan_dos_ids(tmp_path):
    cfg = _cfg(tmp_path)
    tr = CostTracker(cfg.costs_db)
    s = procesar_clip(_video(tmp_path), "c1", cfg, tr,
                      detector=DetectorFalso(n_personas=2))
    tr.close()
    assert s.n_personas_unicas == 2
    assert s.max_personas_simultaneas == 2


def test_cada_etapa_se_registra_en_su_recurso(tmp_path):
    """El Nivel 1 gasta CPU y el Nivel 2 GPU: mezclarlos esconde cual hay que pagar."""
    cfg = _cfg(tmp_path)
    tr = CostTracker(cfg.costs_db)
    s = procesar_clip(_video(tmp_path), "c1", cfg, tr, detector=DetectorFalso(gpu_ms=12.0))
    resumen = tr.resumen_por_stage()
    tr.close()

    assert set(resumen) == {"motion_detection", "person_detection", "tracking", "conteo"}
    # El Nivel 1 no toca la GPU y el Nivel 2 no se factura como CPU.
    assert resumen["motion_detection"]["cpu_total_ms"] > 0
    assert resumen["motion_detection"]["gpu_total_ms"] == 0
    assert resumen["person_detection"]["gpu_total_ms"] == pytest.approx(
        12.0 * s.frames_analizados, rel=1e-6)
    assert resumen["person_detection"]["cpu_total_ms"] == 0

    # El conteo por linea es CPU, no GPU: es geometria sobre cajas.
    assert resumen["conteo"]["gpu_total_ms"] == 0

    con = sqlite3.connect(cfg.costs_db)
    n = con.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0]
    con.close()
    assert n == 4


def test_el_costo_en_dolares_suma_cpu_y_gpu_por_separado(tmp_path):
    tr = CostTracker(tmp_path / "c.db", cpu_usd_per_hour=3600.0, gpu_usd_per_hour=7200.0)
    ev = tr.registrar("mixta", cpu_time_ms=1000.0, gpu_time_ms=1000.0)
    tr.close()
    # 1 s de cpu a 1 usd/s + 1 s de gpu a 2 usd/s
    assert ev.cost_usd == pytest.approx(1.0 + 2.0)


def test_clip_inexistente_no_revienta(tmp_path):
    cfg = _cfg(tmp_path)
    tr = CostTracker(cfg.costs_db)
    s = procesar_clip(tmp_path / "no_existe.mp4", "x", cfg, tr, detector=DetectorFalso())
    tr.close()
    assert "error" in s.meta
    assert s.total_frames == 0


def test_registrar_persiste_un_evento_ya_medido(tmp_path):
    tr = CostTracker(tmp_path / "c.db", gpu_usd_per_hour=3600.0)
    ev = tr.registrar("gpu_stage", gpu_time_ms=250.0, tokens_used=512, clip_id="z")
    tr.close()
    assert ev.gpu_time_ms == 250.0
    assert ev.tokens_used == 512

    con = sqlite3.connect(tmp_path / "c.db")
    fila = con.execute(
        "SELECT stage, cpu_time_ms, gpu_time_ms, tokens_used FROM cost_events"
    ).fetchone()
    con.close()
    assert fila == ("gpu_stage", 0.0, 250.0, 512)


def test_el_stream_corre_el_mismo_cascade_que_un_archivo(tmp_path):
    """procesar_stream estaba ROTO y ningun test lo cubria.

    Era una copia de procesar_clip; al agregarle el conteo por linea a una y no a
    la otra, procesar_stream quedo llamando a un `contador` inexistente y reventaba
    con NameError en el primer frame con movimiento. Ahora las dos comparten bucle,
    pero el test queda para que no vuelva a pasar por otra via.
    """
    cfg = _cfg(tmp_path)
    # procesar_stream abre con CAP_FFMPEG, que tambien lee archivos: se le pasa
    # la ruta como si fuera la url y ejercita el mismo camino.
    ruta = _video(tmp_path)
    tr = CostTracker(cfg.costs_db)
    s = procesar_stream(str(ruta), cfg, tr, detector=DetectorFalso(), max_frames=90)
    resumen = tr.resumen_por_stage()
    tr.close()

    assert "error" not in s.meta
    assert s.frames_analizados > 0
    assert s.frames_descartados > 0
    # Las cuatro etapas, igual que en un archivo: si falta alguna, se desincronizaron.
    assert set(resumen) == {"motion_detection", "person_detection", "tracking", "conteo"}
    assert s.cpu_ms_conteo > 0.0


def test_archivo_y_stream_dan_lo_mismo_sobre_el_mismo_video(tmp_path):
    """Son el mismo cascade: si difieren, una de las dos ramas se quedo atras."""
    cfg = _cfg(tmp_path)
    ruta = _video(tmp_path)
    tr = CostTracker(cfg.costs_db)
    a = procesar_clip(ruta, "c1", cfg, tr, detector=DetectorFalso())
    b = procesar_stream(str(ruta), cfg, tr, detector=DetectorFalso(), max_frames=10**6)
    tr.close()

    assert (a.total_frames, a.frames_analizados, a.frames_descartados) == \
           (b.total_frames, b.frames_analizados, b.frames_descartados)
    assert a.n_personas_unicas == b.n_personas_unicas
    assert a.entradas == b.entradas and a.salidas == b.salidas


def test_max_frames_corta_donde_dice(tmp_path):
    """Antes el corte iba despues de incrementar el contador y se pasaba por uno."""
    cfg = _cfg(tmp_path)
    tr = CostTracker(cfg.costs_db)
    s = procesar_clip(_video(tmp_path, n_frames=90), "c1", cfg, tr,
                      detector=DetectorFalso(), max_frames=25)
    tr.close()
    assert s.total_frames == 25


# ---- Nivel 3 en el pipeline -------------------------------------------------

def test_el_nivel3_se_puede_apagar(tmp_path):
    """Con AMTA_REID=0 el cascade tiene que seguir corriendo igual, sin la etapa."""
    cfg = _cfg(tmp_path, reid=False)
    tr = CostTracker(cfg.costs_db)
    s = procesar_clip(_video(tmp_path), "c1", cfg, tr, detector=DetectorFalso())
    resumen = tr.resumen_por_stage()
    tr.close()

    assert s.frames_analizados > 0
    assert s.gpu_ms_reid == 0.0 and s.n_reenganches == 0
    assert "reid" not in resumen  # no se registra una etapa que no corrio


def test_el_nivel3_se_registra_como_etapa_de_gpu(tmp_path, monkeypatch):
    """El ReID corre en GPU, igual que el detector: no puede facturarse como CPU."""
    import core.cascade.stage2_person.runner as runner

    class ReidFalso:
        """Reengancha todo al id 1 y dice haber gastado 2 ms de GPU por frame."""

        def __init__(self, *a, **kw):
            self.gpu_time_ms = 0.0
            self.n_reenganches = 0

        def procesar(self, frame, dets):
            self.gpu_time_ms = 2.0
            for d in dets:
                if d.track_id not in (None, 1):
                    self.n_reenganches += 1
                d.track_id = 1
            return dets

    monkeypatch.setattr(runner, "ReIdentificador", ReidFalso)
    cfg = _cfg(tmp_path, reid=True)
    tr = CostTracker(cfg.costs_db)
    s = procesar_clip(_video(tmp_path), "c1", cfg, tr,
                      detector=DetectorFalso(n_personas=2))
    resumen = tr.resumen_por_stage()
    tr.close()

    assert "reid" in resumen
    assert resumen["reid"]["gpu_total_ms"] > 0
    assert resumen["reid"]["cpu_total_ms"] == 0
    # Las dos personas del doble quedan fusionadas en una por el ReID falso: eso
    # confirma que el Nivel 3 corre ANTES del registro y que este cuenta sobre sus ids.
    assert s.n_personas_unicas == 1
    assert s.n_reenganches > 0


def test_la_cpu_se_mide_por_hilo_y_no_por_proceso(tmp_path):
    """process_time cuenta los hilos de torch y contamina la medida del Nivel 1.

    Se vio con el mismo trabajo dando 133 s y 153 s segun cuanto corriera el
    detector en paralelo. El test fija que un hilo quemando CPU al lado no se
    sume al coste del clip.
    """
    import threading

    parar = threading.Event()

    def quemar():
        x = 0
        while not parar.is_set():
            x += 1

    ruido = threading.Thread(target=quemar, daemon=True)
    cfg = _cfg(tmp_path)
    tr = CostTracker(cfg.costs_db)
    limpio = procesar_clip(_video(tmp_path), "c1", cfg, tr, detector=DetectorFalso())
    ruido.start()
    try:
        sucio = procesar_clip(_video(tmp_path), "c2", cfg, tr, detector=DetectorFalso())
    finally:
        parar.set()
        ruido.join(timeout=2)
    tr.close()

    # Mismo trabajo: el hilo de ruido no puede duplicar el coste atribuido.
    assert sucio.cpu_ms_motion < limpio.cpu_ms_motion * 1.8
