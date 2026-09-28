"""Tests de la API REST.

Corren SIN GPU y SIN pesos: el motor se sustituye por uno que inyecta un detector
falso. Eso no los vuelve tests de mentira — el video es real, MOG2 corre de
verdad, el seguimiento, el conteo y la deteccion de filas tambien. Lo unico
simulado es de donde salen las cajas, que es justamente la parte que la API no
implementa.

La cobertura sigue la lista pedida:
  - procesamiento correcto de un video          -> test_video_*
  - procesamiento correcto de un stream RTSP    -> test_rtsp_*
  - parametros invalidos                        -> TestValidacion
  - video inexistente                           -> test_clip_inexistente, test_archivo_*
  - modelo no disponible                        -> test_modelo_no_disponible
  - job_id inexistente                          -> TestJobInexistente
  - estructura de las respuestas                -> TestEstructura
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.api.esquemas import ResultadoProceso
from core.api.main import app, get_motor
from core.api.servicio import Motor
from core.cascade.config import load_cascade_config
from core.types import Detection

ANCHO, ALTO = 320, 240
N_FRAMES = 90
QUIETOS = 35  # MOG2 necesita 30 de warmup: antes de eso nada llega al Nivel 2


# --------------------------------------------------------------------------
# Andamios
# --------------------------------------------------------------------------

class DetectorFalso:
    """Devuelve una caja que se desplaza. No carga pesos ni toca la GPU.

    Cumple la interfaz que el runner usa del detector: `track`, `gpu_time_ms`,
    `reset` y `calentar`. Si esa interfaz cambia, este test se rompe, que es
    exactamente lo que se quiere.
    """

    def __init__(self) -> None:
        self.gpu_time_ms = 0.0
        self.n = 0

    def track(self, frame, prompts):
        self.n += 1
        x = 20 + (self.n * 3) % (ANCHO - 80)
        self.gpu_time_ms = 0.5
        return [Detection(bbox=(float(x), 90.0, float(x + 50), 190.0),
                          label="person", score=0.9, track_id=1)]

    detect = track

    def reset(self) -> None:
        self.n = 0

    def calentar(self, alto=480, ancho=640) -> float:
        return 0.0


def escribir_video(path, n_frames=N_FRAMES, quietos=QUIETOS):
    """Video con un rectangulo que se mueve despues del warmup de MOG2."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (ANCHO, ALTO))
    for i in range(n_frames):
        f = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
        f[:] = 30
        if i >= quietos:
            x = 20 + (i - quietos) * 3
            f[90:190, x:x + 50] = 235
        vw.write(f)
    vw.release()
    return path


class MotorDePrueba(Motor):
    """Motor con detector falso y modelo siempre disponible."""

    def __init__(self, cfg, detector=None):
        super().__init__(cfg=cfg)
        self.falso = detector or DetectorFalso()

    def verificar_modelo(self, params=None):
        return True, "detector falso"

    def _detector(self, cfg):
        return self.falso


class MotorSinModelo(MotorDePrueba):
    """El detector no esta: es lo que pasa si faltan los pesos o no hay CUDA."""

    def verificar_modelo(self, params=None):
        return False, "faltan los pesos del backend 'yolo11': /opt/amta/models/yolo11n-seg.pt"


@pytest.fixture
def cfg_tmp(tmp_path):
    """Config apuntada a directorios temporales: los tests no tocan el repo."""
    cfg = load_cascade_config()
    cfg.corpus_dir = tmp_path / "corpus"
    cfg.raw_dir = cfg.corpus_dir / "raw"
    cfg.results_dir = tmp_path / "results"
    cfg.costs_db = cfg.results_dir / "costs.db"
    cfg.raw_dir.mkdir(parents=True)
    cfg.results_dir.mkdir(parents=True)
    return cfg


@pytest.fixture
def video(cfg_tmp):
    return escribir_video(cfg_tmp.raw_dir / "prueba.mp4")


@pytest.fixture
def motor(cfg_tmp, video):
    m = MotorDePrueba(cfg_tmp)
    yield m
    m.cerrar()


@pytest.fixture
def cliente(motor):
    app.dependency_overrides[get_motor] = lambda: motor
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def esperar(cliente, job_id, timeout=60.0):
    """Consulta el estado hasta que sea terminal. Devuelve el cuerpo final."""
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        r = cliente.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        if r.json()["status"] in ("completed", "failed", "cancelled"):
            return r.json()
        time.sleep(0.05)
    pytest.fail(f"el job {job_id} no termino en {timeout}s")


def lanzar(cliente, **extra):
    """POST /jobs sobre el clip de prueba, sin ReID (necesitaria torch)."""
    cuerpo = {"fuente": {"tipo": "clip", "valor": "prueba"}, "usar_reid": False}
    cuerpo.update(extra)
    return cliente.post("/jobs", json=cuerpo)


# --------------------------------------------------------------------------
# Procesamiento de un video
# --------------------------------------------------------------------------

def test_video_se_procesa_completo(cliente):
    r = lanzar(cliente)
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "queued"  # el 202 siempre reporta el estado al aceptar

    final = esperar(cliente, job["job_id"])
    assert final["status"] == "completed", final.get("error")
    assert final["error"] is None

    res = cliente.get(f"/jobs/{job['job_id']}/resultados").json()
    assert res["performance"]["frames_total"] == N_FRAMES
    # El filtro de movimiento tiene que haber hecho algo en las dos direcciones:
    # descartar los frames quietos y dejar pasar los del rectangulo.
    assert res["frames_processed"] > 0
    assert res["frames_discarded"] > 0
    assert res["people_count"] >= 1
    assert res["processing_time"] > 0


def test_video_sin_filtro_analiza_todos_los_frames(cliente):
    """usar_filtro_movimiento=False es la corrida de control del punto 7."""
    job = lanzar(cliente, usar_filtro_movimiento=False).json()
    esperar(cliente, job["job_id"])
    res = cliente.get(f"/jobs/{job['job_id']}/resultados").json()
    assert res["frames_discarded"] == 0
    assert res["frames_processed"] == N_FRAMES


def test_video_respeta_max_frames(cliente):
    job = lanzar(cliente, max_frames=40).json()
    esperar(cliente, job["job_id"])
    res = cliente.get(f"/jobs/{job['job_id']}/resultados").json()
    assert res["performance"]["frames_total"] == 40


def test_video_con_roi_reporta_fila(cliente):
    """Con ROI declarada el bloque queue deja de ser un placeholder."""
    job = lanzar(cliente, roi={"puntos": [[0.0, 0.2], [1.0, 0.2], [1.0, 0.95], [0.0, 0.95]],
                               "min_personas": 1, "dwell_min_s": 0.1}).json()
    esperar(cliente, job["job_id"])
    res = cliente.get(f"/jobs/{job['job_id']}/resultados").json()
    assert res["queue"]["roi_defined"] is True
    assert res["queue"]["max_in_zone"] >= 1


def test_sin_roi_la_fila_no_se_evalua(cliente):
    job = lanzar(cliente).json()
    esperar(cliente, job["job_id"])
    res = cliente.get(f"/jobs/{job['job_id']}/resultados").json()
    assert res["queue"]["roi_defined"] is False
    assert res["queue_detected"] is False


def test_subir_video_y_procesarlo(cliente, tmp_path):
    subido = escribir_video(tmp_path / "subido.mp4")
    with open(subido, "rb") as f:
        r = cliente.post("/jobs/video", files={"archivo": ("subido.mp4", f, "video/mp4")},
                         data={"peticion": '{"usar_reid": false}'})
    assert r.status_code == 202, r.text
    final = esperar(cliente, r.json()["job_id"])
    assert final["status"] == "completed", final.get("error")


def test_subir_video_demasiado_grande(cliente, motor, monkeypatch):
    """Sin tope, una sola peticion llena el disco del servidor.

    Y ahi no falla solo el job: falla todo lo que necesite escribir, empezando
    por costs.db, que es donde vive la medicion de coste.
    """
    monkeypatch.setattr("core.api.servicio.MAX_SUBIDA_BYTES", 1024)
    r = cliente.post("/jobs/video",
                     files={"archivo": ("grande.mp4", b"\0" * 8192, "video/mp4")})
    assert r.status_code == 413
    assert r.json()["codigo"] == "subida_demasiado_grande"
    # Y no queda un archivo a medias ocupando lugar.
    assert not list(motor.dir_subidas.iterdir())


def test_purga_de_subidas_viejas(cliente, motor, tmp_path):
    """Los videos subidos se borran solos: si no, el directorio no se vacia nunca."""
    import os

    motor.dir_subidas.mkdir(parents=True, exist_ok=True)
    viejo = motor.dir_subidas / "viejo.mp4"
    viejo.write_bytes(b"x" * 10)
    os.utime(viejo, (0, 0))  # epoch: mas viejo que cualquier retencion
    reciente = motor.dir_subidas / "reciente.mp4"
    reciente.write_bytes(b"x" * 10)

    assert motor.purgar_subidas(horas=24) == 1
    assert not viejo.exists()
    assert reciente.exists()  # el reciente no se toca


def test_subir_archivo_que_no_es_video(cliente):
    r = cliente.post("/jobs/video",
                     files={"archivo": ("notas.txt", b"esto no es un video", "text/plain")})
    assert r.status_code == 400
    assert r.json()["codigo"] == "fuente_invalida"


# --------------------------------------------------------------------------
# Procesamiento de un stream RTSP
# --------------------------------------------------------------------------

def test_rtsp_se_procesa(cliente, video, monkeypatch):
    """El camino de procesar_stream, con la captura apuntada al archivo local.

    Lo que se prueba es el codigo del servicio: que una fuente rtsp entre por
    procesar_stream, respete el tope y produzca la misma estructura de resultado.
    El transporte RTSP en si es de mediamtx y se verifica levantando el stack, no
    en un test unitario.
    """
    from core.cascade.stage2_person import runner

    real = cv2.VideoCapture

    def captura_falsa(fuente, *a, **kw):
        return real(str(video)) if str(fuente).startswith("rtsp") else real(fuente, *a, **kw)

    monkeypatch.setattr(runner.cv2, "VideoCapture", captura_falsa)

    r = cliente.post("/jobs", json={
        "fuente": {"tipo": "rtsp", "valor": "rtsp://mediamtx:8554/cam1"},
        "usar_reid": False, "max_frames": 60})
    assert r.status_code == 202, r.text
    final = esperar(cliente, r.json()["job_id"])
    assert final["status"] == "completed", final.get("error")

    res = cliente.get(f"/jobs/{r.json()['job_id']}/resultados").json()
    assert res["source_type"] == "rtsp"
    assert res["performance"]["frames_total"] == 60
    ResultadoProceso.model_validate(res)


def test_rtsp_inalcanzable_falla_con_motivo(cliente):
    """Puerto cerrado a proposito. El job tiene que fallar, no colgarse."""
    r = cliente.post("/jobs", json={
        "fuente": {"tipo": "rtsp", "valor": "rtsp://127.0.0.1:9/nada"},
        "usar_reid": False, "max_frames": 10, "max_segundos": 15})
    assert r.status_code == 202
    final = esperar(cliente, r.json()["job_id"], timeout=90.0)
    assert final["status"] == "failed"
    assert "rtsp://127.0.0.1:9/nada" in final["error"]


def test_rtsp_sin_tope_recibe_uno_por_defecto():
    """Un stream no termina solo: sin tope el job no acabaria nunca.

    Se verifica sobre el esquema y no por HTTP a proposito: mandarlo de verdad
    dejaria un job intentando abrir un stream inexistente durante todo el resto
    de la suite.
    """
    from core.api.esquemas import PeticionProceso

    p = PeticionProceso.model_validate(
        {"fuente": {"tipo": "rtsp", "valor": "rtsp://host:8554/cam1"}})
    assert p.max_segundos == 60.0

    # Si el cliente si puso un tope, se respeta el suyo.
    q = PeticionProceso.model_validate(
        {"fuente": {"tipo": "rtsp", "valor": "rtsp://host:8554/cam1"},
         "max_frames": 10})
    assert q.max_segundos == 0.0 and q.max_frames == 10


class CapturaInterminable:
    """Envuelve una captura para que nunca se acabe y avance despacio.

    Es un wrapper y no un monkeypatch del metodo: cv2.VideoCapture es un tipo de
    C y a sus instancias no se les pueden asignar atributos. Simula lo unico que
    importa de una camara en vivo para este test: que `read` no devuelve False
    nunca, asi que el unico final posible es la cancelacion.
    """

    def __init__(self, cap):
        self._cap = cap

    def isOpened(self):
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)

    def read(self):
        time.sleep(0.02)
        ok, f = self._cap.read()
        return (True, f if ok else np.zeros((ALTO, ANCHO, 3), np.uint8))

    def release(self):
        self._cap.release()


def test_max_segundos_termina_completed_no_cancelled(cliente, video, monkeypatch):
    """Agotar el tope que pidio el cliente es terminar bien, no cancelarse.

    Solo un DELETE deja el job en `cancelled`. El runner no distingue los dos
    casos (para el bucle las dos son una parada externa); el motor si.
    """
    from core.cascade.stage2_person import runner

    real = cv2.VideoCapture

    def captura_lenta(fuente, *a, **kw):
        if str(fuente).startswith("rtsp"):
            return CapturaInterminable(real(str(video)))
        return real(fuente, *a, **kw)

    monkeypatch.setattr(runner.cv2, "VideoCapture", captura_lenta)

    r = cliente.post("/jobs", json={
        "fuente": {"tipo": "rtsp", "valor": "rtsp://host:8554/cam1"},
        "usar_reid": False, "max_segundos": 1.0})
    final = esperar(cliente, r.json()["job_id"], timeout=30.0)
    assert final["status"] == "completed", final
    assert cliente.get(f"/jobs/{r.json()['job_id']}/resultados").json()["frames_processed"] > 0


def test_cancelar_job(cliente, video, monkeypatch):
    """DELETE es la unica forma de terminar un stream en vivo sin tope."""
    from core.cascade.stage2_person import runner

    real = cv2.VideoCapture

    def captura_lenta(fuente, *a, **kw):
        if str(fuente).startswith("rtsp"):
            return CapturaInterminable(real(str(video)))
        return real(fuente, *a, **kw)

    monkeypatch.setattr(runner.cv2, "VideoCapture", captura_lenta)

    r = cliente.post("/jobs", json={
        "fuente": {"tipo": "rtsp", "valor": "rtsp://host:8554/cam1"},
        "usar_reid": False, "max_segundos": 120})
    job_id = r.json()["job_id"]
    # Se espera a que `progress` avance y no solo a que el estado sea "running".
    # El worker marca running antes de abrir la captura: cancelando ahi, la
    # cancelacion llega antes del primer frame y no habria nada que contabilizar.
    # De paso, esto es lo unico que verifica que el progreso se publique en vivo.
    limite = time.monotonic() + 20
    while True:
        estado = cliente.get(f"/jobs/{job_id}").json()
        if estado["progress"]["frames_read"] > 0:
            break
        assert estado["status"] in ("queued", "running"), estado
        # people_so_far tiene que subir DURANTE la corrida, no quedarse en cero
        # hasta el final: el detector falso pone una persona desde el frame 1.
        if estado["progress"]["frames_processed"] > 5:
            assert estado["progress"]["people_so_far"] > 0, estado
        if time.monotonic() > limite:
            pytest.fail(f"el job nunca avanzo: {estado}")
        time.sleep(0.05)

    assert cliente.delete(f"/jobs/{job_id}").status_code == 200
    final = esperar(cliente, job_id, timeout=30.0)
    assert final["status"] == "cancelled"
    # Lo procesado antes de cancelar se pago y tiene que poder contabilizarse.
    m = cliente.get(f"/jobs/{job_id}/metricas")
    assert m.status_code == 200
    assert m.json()["performance"]["frames_total"] > 0


# --------------------------------------------------------------------------
# Fuentes que no existen
# --------------------------------------------------------------------------

def test_clip_inexistente(cliente):
    r = cliente.post("/jobs", json={"fuente": {"tipo": "clip", "valor": "no_existe"}})
    assert r.status_code == 404
    assert r.json()["codigo"] == "fuente_no_encontrada"


def test_archivo_inexistente(cliente, cfg_tmp):
    r = cliente.post("/jobs", json={
        "fuente": {"tipo": "archivo", "valor": str(cfg_tmp.raw_dir / "fantasma.mp4")}})
    assert r.status_code == 404


def test_archivo_fuera_del_area_servida(cliente):
    """Una API que abre cualquier ruta lee cualquier video del disco del servidor."""
    r = cliente.post("/jobs", json={"fuente": {"tipo": "archivo", "valor": "/etc/hostname"}})
    assert r.status_code == 400
    assert r.json()["codigo"] == "fuente_invalida"


def test_clip_id_con_separadores_es_rechazado(cliente):
    r = cliente.post("/jobs", json={"fuente": {"tipo": "clip", "valor": "../../etc/passwd"}})
    assert r.status_code == 422


# --------------------------------------------------------------------------
# Modelo no disponible
# --------------------------------------------------------------------------

def test_modelo_no_disponible(cfg_tmp, video):
    m = MotorSinModelo(cfg_tmp)
    app.dependency_overrides[get_motor] = lambda: m
    try:
        with TestClient(app) as c:
            r = c.post("/jobs", json={"fuente": {"tipo": "clip", "valor": "prueba"}})
            assert r.status_code == 503
            assert r.json()["codigo"] == "modelo_no_disponible"
            assert "yolo11n-seg.pt" in r.json()["detail"]

            salud = c.get("/salud").json()
            assert salud["status"] == "degraded"
            assert salud["model_available"] is False
    finally:
        app.dependency_overrides.clear()
        m.cerrar()


def test_salud_con_modelo_disponible(cliente):
    s = cliente.get("/salud").json()
    assert s["status"] == "ok"
    assert s["model_available"] is True


# --------------------------------------------------------------------------
# job_id inexistente
# --------------------------------------------------------------------------

class TestJobInexistente:
    """Un id bien formado pero desconocido es 404; uno imposible de emitir, 422."""

    @pytest.mark.parametrize("ruta", ["/jobs/{}", "/jobs/{}/resultados", "/jobs/{}/metricas"])
    def test_404_en_todas_las_consultas(self, cliente, ruta):
        r = cliente.get(ruta.format("abc123"))
        assert r.status_code == 404
        assert "abc123" in r.json()["detail"]

    def test_404_al_cancelar(self, cliente):
        assert cliente.delete("/jobs/abc123").status_code == 404

    @pytest.mark.parametrize("malo", ["con espacio", "x" * 65, "dos%puntos"])
    def test_422_si_el_id_es_imposible(self, cliente, malo):
        assert cliente.get(f"/jobs/{malo}").status_code == 422

    def test_409_si_todavia_no_termino(self, cliente, motor):
        from core.api.esquemas import PeticionProceso

        job = motor.almacen.crear(PeticionProceso.model_validate(
            {"fuente": {"tipo": "clip", "valor": "prueba"}}))
        r = cliente.get(f"/jobs/{job.job_id}/resultados")
        assert r.status_code == 409
        assert "queued" in r.json()["detail"]


# --------------------------------------------------------------------------
# Parametros invalidos
# --------------------------------------------------------------------------

class TestValidacion:
    """Todo lo que Pydantic tiene que rechazar antes de tocar el pipeline."""

    @pytest.mark.parametrize("modelo", [
        {"conf": 1.5},                  # fuera de [0,1]
        {"conf": -0.1},
        {"iou": 2.0},
        {"imgsz": 641},                 # no es multiplo de 32
        {"imgsz": 64},                  # por debajo del minimo
        {"imgsz": 4096},
        {"backend": "yolov3"},          # backend que no existe
        {"device": "gpu0"},             # ni cpu, ni un indice, ni cuda:N
        {"prompts": []},                # sin nada que buscar
        {"prompts": ["  "]},
        {"umbral": 0.5},                # campo desconocido: extra="forbid"
    ])
    def test_parametros_de_modelo(self, cliente, modelo):
        r = cliente.post("/jobs", json={"fuente": {"tipo": "clip", "valor": "prueba"},
                                        "modelo": modelo})
        assert r.status_code == 422, f"{modelo} deberia ser rechazado"

    @pytest.mark.parametrize("fuente", [
        {"tipo": "rtsp", "valor": "http://host/cam"},   # rtsp que no es rtsp
        {"tipo": "rtsp", "valor": "cam1"},
        {"tipo": "camara", "valor": "0"},               # tipo inexistente
        {"tipo": "clip", "valor": ""},
        {"tipo": "clip", "valor": "sub/dir"},
        {"valor": "prueba"},                            # falta el tipo
        {"tipo": "clip"},                               # falta el valor
    ])
    def test_fuente(self, cliente, fuente):
        assert cliente.post("/jobs", json={"fuente": fuente}).status_code == 422

    @pytest.mark.parametrize("roi", [
        {"puntos": [[0.1, 0.1], [0.9, 0.1]]},                       # dos puntos no es poligono
        {"puntos": [[0.1, 0.1], [0.9, 0.1], [1.5, 0.9]]},           # fuera de [0,1]
        {"puntos": [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]},           # vertices repetidos
        {"puntos": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]], "min_personas": 0},
        {"puntos": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]], "dwell_min_s": -1},
        {"puntos": [[0, 0, 0], [0.9, 0.1], [0.9, 0.9]]},            # punto de 3 coordenadas
    ])
    def test_roi(self, cliente, roi):
        r = cliente.post("/jobs", json={"fuente": {"tipo": "clip", "valor": "prueba"},
                                        "roi": roi})
        assert r.status_code == 422, f"{roi} deberia ser rechazada"

    @pytest.mark.parametrize("conteo", [
        {"linea": [[0.35, 0.0], [0.35, 0.0]]},   # los dos extremos iguales
        {"linea": [[0.35, 0.0]]},                # una linea necesita dos puntos
        {"linea": [[0.35, 0.0], [1.4, 1.0]]},    # fuera del frame
    ])
    def test_linea_de_conteo(self, cliente, conteo):
        r = cliente.post("/jobs", json={"fuente": {"tipo": "clip", "valor": "prueba"},
                                        "conteo": conteo})
        assert r.status_code == 422

    @pytest.mark.parametrize("extra", [
        {"max_frames": -1}, {"max_segundos": -5}, {"max_segundos": 999_999},
        {"usar_reid": "quiza"}, {"campo_inventado": 1},
    ])
    def test_resto_de_la_peticion(self, cliente, extra):
        cuerpo = {"fuente": {"tipo": "clip", "valor": "prueba"}}
        cuerpo.update(extra)
        assert cliente.post("/jobs", json=cuerpo).status_code == 422

    def test_cuerpo_vacio(self, cliente):
        assert cliente.post("/jobs", json={}).status_code == 422

    def test_el_error_422_dice_que_campo_fallo(self, cliente):
        r = cliente.post("/jobs", json={"fuente": {"tipo": "clip", "valor": "prueba"},
                                        "modelo": {"conf": 1.5}})
        assert "conf" in str(r.json()["detail"])


# --------------------------------------------------------------------------
# Estructura de las respuestas
# --------------------------------------------------------------------------

class TestEstructura:
    """El contrato: nombres, tipos y que ningun campo se caiga al serializar."""

    CONTRATO = {
        "job_id": str, "status": str, "people_count": int, "queue_detected": bool,
        "queue_size": int, "frames_processed": int, "frames_discarded": int,
        "processing_time": float, "estimated_cost": float,
    }

    @pytest.fixture
    def resultado(self, cliente):
        job = lanzar(cliente).json()
        esperar(cliente, job["job_id"])
        return cliente.get(f"/jobs/{job['job_id']}/resultados").json()

    def test_campos_del_contrato(self, resultado):
        for campo, tipo in self.CONTRATO.items():
            assert campo in resultado, f"falta {campo}"
            # bool es subclase de int en Python: hay que mirarlo primero.
            if tipo is bool:
                assert isinstance(resultado[campo], bool), campo
            elif tipo is float:
                assert isinstance(resultado[campo], (int, float)), campo
            else:
                assert isinstance(resultado[campo], tipo), campo

    def test_bloques_anidados(self, resultado):
        for bloque in ("detection", "counting", "queue", "performance", "cost"):
            assert isinstance(resultado[bloque], dict), bloque
        assert set(resultado["counting"]) == {"entered", "exited", "crossed_line"}
        assert resultado["cost"]["cpu_ms_total"] >= 0
        assert "per_stage" in resultado["cost"]

    def test_valida_contra_el_esquema(self, resultado):
        ResultadoProceso.model_validate(resultado)

    def test_coherencia_interna(self, resultado):
        p = resultado["performance"]
        assert p["frames_processed"] == resultado["frames_processed"]
        assert p["frames_discarded"] == resultado["frames_discarded"]
        assert p["frames_total"] >= p["frames_processed"] + p["frames_discarded"]
        assert resultado["queue_size"] == resultado["queue"]["size"]

    def test_estado_del_job(self, cliente):
        job = lanzar(cliente).json()
        assert set(job) == {"job_id", "status", "source_type", "source", "created_at",
                            "started_at", "finished_at", "progress", "error"}
        assert set(job["progress"]) == {"frames_read", "frames_processed",
                                        "frames_discarded", "people_so_far", "elapsed_s"}

    def test_listado(self, cliente):
        lanzar(cliente)
        r = cliente.get("/jobs")
        assert r.status_code == 200
        assert isinstance(r.json(), list) and r.json()
        assert cliente.get("/jobs", params={"status": "completed"}).status_code == 200
        assert cliente.get("/jobs", params={"limite": 0}).status_code == 422

    def test_metricas_globales(self, cliente):
        job = lanzar(cliente).json()
        esperar(cliente, job["job_id"])
        m = cliente.get("/metricas").json()
        assert m["jobs_total"] >= 1
        assert m["frames_total"] >= N_FRAMES
        assert m["jobs_by_status"]["completed"] >= 1
        # El log de coste tiene una fila por etapa del cascade.
        assert {"motion_detection", "person_detection"} <= set(m["per_stage"])

    def test_metricas_del_job(self, cliente):
        job = lanzar(cliente).json()
        esperar(cliente, job["job_id"])
        m = cliente.get(f"/jobs/{job['job_id']}/metricas").json()
        assert m["job_id"] == job["job_id"]
        assert m["performance"]["fps"] > 0
        assert m["cost"]["gpu_ms_total"] >= 0

    def test_openapi_se_genera(self, cliente):
        """Si el esquema no se puede generar, /docs queda roto sin avisar."""
        r = cliente.get("/openapi.json")
        assert r.status_code == 200
        rutas = r.json()["paths"]
        for ruta in ("/jobs", "/jobs/{job_id}", "/jobs/{job_id}/resultados",
                     "/jobs/{job_id}/metricas", "/metricas", "/salud"):
            assert ruta in rutas, ruta
