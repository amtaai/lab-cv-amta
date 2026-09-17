"""Corre el cascade completo sobre un clip: Nivel 1 -> Nivel 2 -> registro de tracks.

Aca la arquitectura en cascada existe de verdad:
  - MOG2 corre sobre el 100 % de los frames (es barato, CPU).
  - El detector corre SOLO sobre los frames que MOG2 dejo pasar (es caro, GPU).
  - ByteTrack va adentro del detector, asi que el track_id llega con la caja.

El detector es el de `core/perception/detector.py` — el mismo que usan los
notebooks de `yolo_seg/` y `yolo_world/`. Este modulo no implementa deteccion ni
seguimiento: los cablea al filtro de movimiento y les mide el costo.

Las etapas se miden por separado y se registran con tracker.registrar() en vez de
tracker.track(): corren INTERCALADAS dentro del mismo loop, asi que sus bloques se
solapan y anidar context managers haria que cada uno midiera el tiempo del otro.

El Nivel 1 se mide en CPU y el Nivel 2 en GPU (reloj de pared con sincronizacion
CUDA). Son unidades distintas a proposito: son recursos distintos y mezclarlos en
un solo numero esconde cual de los dos es el que hay que pagar.

La CPU se mide con `time.thread_time_ns()` y NO con `process_time_ns()`. La
Semana 1 usaba process_time y anoto que solo valia para etapas secuenciales; al
entrar el detector en GPU eso dejo de cumplirse, porque torch levanta hilos
propios y su CPU se sumaba a la ventana donde se mide MOG2. Se vio midiendo el
mismo trabajo dos veces: 133.268 ms contra 153.621 ms sobre los mismos 5.838
frames, segun cuanto trabajaba el detector en paralelo. thread_time cuenta solo
el hilo que llama, que es donde corre MOG2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from core.cascade.config import CascadeConfig
from core.cascade.cost.tracker import CostTracker
from core.cascade.stage1_motion.detector import MotionDetector
from core.perception.detector import OpenVocabDetector, crear_detector
from core.perception.tracker import Tracker
from core.perception.conteo import ContadorPersonas
from core.perception.filas import DetectorFila, FilaConfig
from core.perception.reid import ReIdConfig, ReIdentificador


def construir_detector(cfg: CascadeConfig) -> OpenVocabDetector:
    """Instancia el detector del proyecto segun la config del cascade."""
    kw = dict(imgsz=cfg.person.imgsz, conf=cfg.person.conf, iou=cfg.person.iou,
              device=cfg.person.device, tracker=cfg.track.tracker)
    if cfg.person.pesos:
        kw["pesos"] = cfg.person.pesos
    return crear_detector(cfg.person.backend, **kw)


@dataclass
class ClipPersonStats:
    """Que vio el Nivel 2 en un clip y cuanto costo verlo."""

    clip_id: str
    total_frames: int = 0
    warmup_frames: int = 0
    frames_analizados: int = 0  # los que el Nivel 1 dejo pasar y vio el Nivel 2
    frames_descartados: int = 0  # los que el Nivel 2 nunca vio: el ahorro
    frames_con_persona: int = 0  # de los analizados, cuantos traian gente
    n_detecciones: int = 0  # cajas totales (una persona en 50 frames son 50)
    n_personas_unicas: int = 0  # tracks confirmados por ByteTrack + el registro
    max_personas_simultaneas: int = 0
    # --- punto 3: conteo por cruce de linea ---
    entradas: int = 0  # cruzaron la linea en direccion IN
    salidas: int = 0  # cruzaron en direccion OUT
    contados_por_linea: int = 0  # tracks distintos que cruzaron alguna vez
    conf_media: float = 0.0
    cpu_ms_motion: float = 0.0
    gpu_ms_person: float = 0.0  # el Nivel 2 corre en GPU
    cpu_ms_tracking: float = 0.0  # el registro de tracks si es CPU
    cpu_ms_conteo: float = 0.0  # el conteo por cruce de linea, tambien CPU
    gpu_ms_reid: float = 0.0  # el Nivel 3 tambien corre en GPU
    n_reenganches: int = 0  # veces que el Nivel 3 recupero un ID perdido
    wall_ms: float = 0.0  # reloj de pared del clip entero: de aca sale el FPS
    # --- punto 4: fila en la zona de espera (solo si el clip tiene ROI) ---
    tiene_roi: bool = False
    max_en_zona: int = 0
    max_en_fila: int = 0
    pct_frames_con_fila: float = 0.0
    hubo_fila: bool = False
    cancelado: bool = False  # lo corto un pedido externo, no el fin de la fuente
    meta: dict = field(default_factory=dict)

    @property
    def frames_utiles(self) -> int:
        """Frames sin contar el warmup de MOG2."""
        return self.frames_analizados + self.frames_descartados

    @property
    def pct_frames_con_persona(self) -> float:
        """Sobre los frames que el Nivel 2 realmente analizo."""
        if not self.frames_analizados:
            return 0.0
        return round(100.0 * self.frames_con_persona / self.frames_analizados, 2)

    @property
    def pct_analizado(self) -> float:
        """Que fraccion del video util llego al Nivel 2."""
        return round(100.0 * self.frames_analizados / self.frames_utiles, 2) \
            if self.frames_utiles else 0.0

    @property
    def fps(self) -> float:
        """Frames de video por segundo de reloj. Es el ritmo real del pipeline."""
        if not self.wall_ms:
            return 0.0
        return round(self.total_frames / (self.wall_ms / 1000.0), 2)

    @property
    def ms_por_frame_analizado(self) -> float:
        """Costo del Nivel 2 por frame que efectivamente analizo, en ms de GPU."""
        if not self.frames_analizados:
            return 0.0
        return round(self.gpu_ms_person / self.frames_analizados, 3)


def _acumular(stats: ClipPersonStats, dets: list) -> float:
    """Suma las detecciones del frame a las stats. Devuelve la suma de scores."""
    if not dets:
        return 0.0
    stats.frames_con_persona += 1
    stats.n_detecciones += len(dets)
    stats.max_personas_simultaneas = max(stats.max_personas_simultaneas, len(dets))
    return sum(d.score for d in dets)


@dataclass
class _Medidas:
    """Lo que el bucle acumula mientras corre. Interno."""

    ns_mov: int = 0
    ns_reg: int = 0
    ns_cnt: int = 0
    ms_gpu: float = 0.0
    ms_reid: float = 0.0
    suma_conf: float = 0.0


def _bucle(cap, stats: ClipPersonStats, cfg: CascadeConfig,
           det_per: OpenVocabDetector, max_frames: int,
           sink=None, usar_gate: bool = True, roi=None, fila_cfg=None,
           debe_parar=None, progreso=None, cada_progreso: int = 30) -> tuple:
    """El cascade sobre una captura ya abierta. Lo unico que hay que mantener.

    `sink` recibe (frame_idx, dets) por cada frame analizado, para exportar sin
    acumular nada en memoria. `usar_gate=False` saltea el Nivel 1 y manda TODOS
    los frames al detector: es la corrida de control contra la que se compara el
    ahorro de la cascada.

    `debe_parar` es un callable sin argumentos que, si devuelve True, corta el
    bucle y marca `stats.cancelado`. Es obligatorio para servir esto por API: un
    stream RTSP no termina nunca, asi que sin una salida externa el unico corte
    posible seria `max_frames`, y eso convierte "cancelar" en "esperar".

    `progreso` recibe una copia de `stats` cada `cada_progreso` frames, para que
    quien consulte el estado del job vea avance y no una caja negra.

    Vive aparte porque archivo y stream RTSP solo se diferencian en como se abre
    la captura y cuando se corta. Cuando eran dos funciones copiadas, agregar el
    conteo por linea a una y no a la otra dejo `procesar_stream` llamando a un
    `contador` que no existia: reventaba con NameError en el primer frame con
    movimiento y ningun test lo cubria.
    """
    det_mov = MotionDetector(cfg.motion)
    reg = Tracker(max_age=cfg.track.max_age, min_hits=cfg.track.min_hits)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    contador = ContadorPersonas(cfg.conteo.linea, ancho, alto)
    fila = (DetectorFila(roi, ancho, alto, fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
                         cfg=fila_cfg or FilaConfig())
            if roi and cfg.filas.activo else None)
    stats.tiene_roi = fila is not None
    # El Nivel 3 va ANTES del registro: reescribe los track_id de ByteTrack a
    # identidades estables, y el registro cuenta sobre esas.
    reid = (ReIdentificador(ReIdConfig(), fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
                            device=cfg.person.device)
            if cfg.reid.activo else None)
    prompts = list(cfg.person.prompts)
    m = _Medidas()

    # ByteTrack guarda estado entre llamadas: sin resetear, la primera persona de
    # esta fuente hereda el ID de la ultima de la anterior.
    if hasattr(det_per, "reset"):
        det_per.reset()

    try:
        while True:
            if max_frames and stats.total_frames >= max_frames:
                break
            if debe_parar is not None and debe_parar():
                stats.cancelado = True
                break
            ok, frame = cap.read()
            if not ok:
                break
            stats.total_frames += 1

            t0 = time.thread_time_ns()
            r_mov = det_mov.process(frame)
            m.ns_mov += time.thread_time_ns() - t0

            if usar_gate:
                if r_mov.is_warmup:
                    stats.warmup_frames += 1
                    continue
                if not r_mov.has_motion:
                    stats.frames_descartados += 1
                    continue

            # --- Nivel 2: solo llegan aca los frames con movimiento ---
            stats.frames_analizados += 1
            dets = det_per.track(frame, prompts)
            m.ms_gpu += getattr(det_per, "gpu_time_ms", 0.0)

            # --- Nivel 3: reengancha los IDs que la oclusion partio ---
            if reid is not None:
                dets = reid.procesar(frame, dets)
                m.ms_reid += reid.gpu_time_ms

            t0 = time.thread_time_ns()
            dets = reg.update(dets)
            m.ns_reg += time.thread_time_ns() - t0

            # Punto 3. Se le pasa una copia del frame porque las soluciones de
            # ultralytics dibujan sobre lo que reciben, y el frame anotado no se
            # usa en el barrido (si en el visualizador y en el vivo).
            t0 = time.thread_time_ns()
            contador.actualizar(frame.copy(), dets)
            if fila is not None:
                fila.actualizar(frame.copy(), dets)
            m.ns_cnt += time.thread_time_ns() - t0

            m.suma_conf += _acumular(stats, dets)
            if sink is not None:
                sink(stats.total_frames - 1, dets)
            if progreso is not None and stats.total_frames % cada_progreso == 0:
                # unique_count se copia aca y no solo al final: quien mira el
                # avance de un job en curso tiene que ver el conteo subir, no un
                # cero fijo hasta que termine.
                stats.n_personas_unicas = reg.unique_count
                progreso(stats)
    finally:
        cap.release()

    stats.entradas = contador.entradas
    stats.salidas = contador.salidas
    stats.contados_por_linea = contador.contados
    stats.n_personas_unicas = reg.unique_count
    if stats.n_detecciones:
        stats.conf_media = round(m.suma_conf / stats.n_detecciones, 4)
    stats.cpu_ms_motion = round(m.ns_mov / 1e6, 3)
    stats.cpu_ms_tracking = round(m.ns_reg / 1e6, 3)
    stats.cpu_ms_conteo = round(m.ns_cnt / 1e6, 3)
    stats.gpu_ms_person = round(m.ms_gpu, 3)
    stats.gpu_ms_reid = round(m.ms_reid, 3)
    stats.n_reenganches = reid.n_reenganches if reid is not None else 0
    if fila is not None:
        stats.max_en_zona = fila.max_en_zona
        stats.max_en_fila = fila.max_en_fila
        stats.pct_frames_con_fila = fila.pct_frames_con_fila
        stats.hubo_fila = fila.hubo_fila
    return stats


def _registrar_costos(cost: CostTracker, stats: ClipPersonStats, cfg: CascadeConfig,
                      fuente: str, wall_ms: float) -> None:
    """Una fila por etapa. Cada una en SU recurso: el Nivel 2 es GPU, el resto CPU."""
    comun = {"clip_id": stats.clip_id, "fuente": fuente}
    cost.registrar("motion_detection", cpu_time_ms=stats.cpu_ms_motion,
                   wall_time_ms=wall_ms, n_frames=stats.total_frames, **comun)
    cost.registrar("person_detection", gpu_time_ms=stats.gpu_ms_person,
                   n_frames=stats.frames_analizados,
                   n_detecciones=stats.n_detecciones,
                   backend=cfg.person.backend, **comun)
    cost.registrar("tracking", cpu_time_ms=stats.cpu_ms_tracking,
                   n_frames=stats.frames_analizados,
                   n_personas=stats.n_personas_unicas,
                   tracker=cfg.track.tracker, **comun)
    cost.registrar("conteo", cpu_time_ms=stats.cpu_ms_conteo,
                   n_frames=stats.frames_analizados,
                   entradas=stats.entradas, salidas=stats.salidas, **comun)
    if stats.gpu_ms_reid or stats.n_reenganches:
        cost.registrar("reid", gpu_time_ms=stats.gpu_ms_reid,
                       n_frames=stats.frames_analizados,
                       n_reenganches=stats.n_reenganches, **comun)


def procesar_clip(path: Path, clip_id: str, cfg: CascadeConfig,
                  cost: CostTracker,
                  detector: OpenVocabDetector | None = None,
                  max_frames: int = 0, sink=None,
                  usar_gate: bool = True, roi=None, fila_cfg=None,
                  debe_parar=None, progreso=None) -> ClipPersonStats:
    """Cascade completo sobre un archivo de video.

    `detector` se inyecta para no recargar los pesos en cada clip del barrido:
    levantar el modelo en GPU cuesta segundos y hacerlo 182 veces domina el
    tiempo total y ensucia el ms/frame.
    """
    stats = ClipPersonStats(clip_id=clip_id)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        stats.meta["error"] = f"no se pudo abrir {path}"
        return stats

    t0 = time.perf_counter_ns()
    _bucle(cap, stats, cfg, detector or construir_detector(cfg), max_frames,
           sink=sink, usar_gate=usar_gate, roi=roi, fila_cfg=fila_cfg,
           debe_parar=debe_parar, progreso=progreso)
    wall = (time.perf_counter_ns() - t0) / 1e6
    stats.wall_ms = round(wall, 3)
    _registrar_costos(cost, stats, cfg, "file", wall)
    return stats


def procesar_stream(url: str, cfg: CascadeConfig, cost: CostTracker,
                    detector: OpenVocabDetector | None = None,
                    max_frames: int = 300, sink=None, roi=None, fila_cfg=None,
                    debe_parar=None, progreso=None) -> ClipPersonStats:
    """Cascade completo sobre un stream RTSP.

    `max_frames` tiene default porque los usos de linea de comandos quieren un
    corte; servido por API se pasa 0 y el corte lo da `debe_parar`.
    """
    stats = ClipPersonStats(clip_id="rtsp_stream")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        stats.meta["error"] = f"no se pudo abrir el stream {url}"
        return stats

    t0 = time.perf_counter_ns()
    _bucle(cap, stats, cfg, detector or construir_detector(cfg), max_frames,
           sink=sink, roi=roi, fila_cfg=fila_cfg,
           debe_parar=debe_parar, progreso=progreso)
    wall = (time.perf_counter_ns() - t0) / 1e6
    stats.wall_ms = round(wall, 3)
    _registrar_costos(cost, stats, cfg, "rtsp", wall)
    return stats
