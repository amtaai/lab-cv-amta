"""El motor: traduce una peticion HTTP en una corrida del cascade y viceversa.

Este modulo es la unica pieza que conoce a la vez el mundo de la API y el del
pipeline. Los endpoints (`main.py`) no importan nada de `core.cascade`, y el
runner no sabe que existe una API. Si manana el pipeline se sirve por gRPC o por
una cola, se reescribe `main.py` y esto queda igual.

La GPU se usa de a un job por vez (`max_workers=1`). No es una limitacion
provisional: el Nivel 2 corre en una sola GPU y dos jobs en paralelo no van al
doble, se pelean la VRAM y ademas arruinan la medicion de ms/frame, que es uno de
los entregables. Los jobs de mas esperan en cola y eso se ve en `status`.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from core.api.esquemas import (
    Conteo,
    Costo,
    Deteccion,
    EstadoJob,
    Fila,
    MetricasGlobales,
    ParametrosModelo,
    PeticionProceso,
    Progreso,
    Rendimiento,
    ResultadoProceso,
    Salud,
    TipoFuente,
)
from core.api.esquemas import RespuestaJob
from core.api.jobs import AlmacenJobs, Job
from core.cascade.catalog.index import cargar_indice
from core.cascade.config import CascadeConfig, load_cascade_config
from core.cascade.cost.tracker import CostTracker
from core.cascade.stage2_person.runner import (
    construir_detector,
    procesar_clip,
    procesar_stream,
)
from core.perception.filas import FilaConfig

MS_POR_HORA = 3_600_000.0
EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_SUBIDA_BYTES = int(os.environ.get("AMTA_API_MAX_SUBIDA_MB", "2048")) * 1024 * 1024
HORAS_RETENCION_SUBIDAS = float(os.environ.get("AMTA_API_RETENCION_H", "24"))


class ErrorApi(Exception):
    """Error con un codigo HTTP asociado. `main.py` lo traduce a la respuesta."""

    http = 500
    codigo = "error_interno"

    def __init__(self, detalle: str) -> None:
        super().__init__(detalle)
        self.detalle = detalle


class FuenteNoEncontrada(ErrorApi):
    """El clip o el archivo pedido no existe."""

    http = 404
    codigo = "fuente_no_encontrada"


class FuenteInvalida(ErrorApi):
    """La fuente existe pero no se puede usar (ruta fuera del area permitida, etc)."""

    http = 400
    codigo = "fuente_invalida"


class SubidaDemasiadoGrande(ErrorApi):
    """El video subido supera el tope. 413 es el codigo que le corresponde."""

    http = 413
    codigo = "subida_demasiado_grande"


class ModeloNoDisponible(ErrorApi):
    """El detector no se puede cargar: faltan pesos, o se pidio GPU y no hay."""

    http = 503
    codigo = "modelo_no_disponible"


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


class Motor:
    """Acepta jobs, los corre en segundo plano y responde por sus resultados."""

    def __init__(self, cfg: CascadeConfig | None = None, max_workers: int = 1,
                 almacen: AlmacenJobs | None = None) -> None:
        self.cfg = cfg or load_cascade_config()
        self.almacen = almacen or AlmacenJobs()
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="cascade-job")
        # Un detector por combinacion de parametros: cargar los pesos en GPU
        # cuesta segundos y hacerlo por job hace que un video corto tarde mas en
        # cargar el modelo que en procesarse.
        self._detectores: dict[tuple, object] = {}
        self._lock_det = threading.Lock()
        self.dir_subidas = self.cfg.results_dir.parent / "uploads"

    # ---------------------------------------------------------------- fuentes

    def _dirs_permitidos(self) -> list[Path]:
        """Unicos lugares de los que la API abre un archivo.

        Una API que abre cualquier ruta que le manden deja que un cliente lea
        cualquier video del disco del servidor. Solo el corpus y el directorio de
        subidas.
        """
        return [self.cfg.raw_dir.resolve(), self.dir_subidas.resolve()]

    def resolver_fuente(self, peticion: PeticionProceso) -> str:
        """Devuelve la ruta o URL concreta que hay que abrir. Valida que exista."""
        f = peticion.fuente
        if f.tipo is TipoFuente.RTSP:
            return f.valor  # abrirlo puede fallar; eso lo decide el worker

        if f.tipo is TipoFuente.CLIP:
            for c in cargar_indice(self.cfg.corpus_dir):
                if c.clip_id == f.valor:
                    ruta = self.cfg.raw_dir / c.filename
                    if not ruta.exists():
                        raise FuenteNoEncontrada(
                            f"el clip '{f.valor}' esta en el indice pero falta "
                            f"su archivo: {c.filename}")
                    return str(ruta)
            # No esta indexado: puede estar suelto en corpus/raw/.
            suelto = next(iter(self.cfg.raw_dir.glob(f"{f.valor}.*")), None)
            if suelto is None:
                raise FuenteNoEncontrada(f"no existe el clip '{f.valor}' en el corpus")
            return str(suelto)

        ruta = Path(f.valor).resolve()
        if not any(ruta == d or d in ruta.parents for d in self._dirs_permitidos()):
            raise FuenteInvalida(
                "la ruta esta fuera de los directorios servidos "
                "(corpus/raw y uploads). Suba el video con POST /jobs/video "
                "o use fuente.tipo='clip'.")
        if not ruta.is_file():
            raise FuenteNoEncontrada(f"no existe el archivo {f.valor}")
        return str(ruta)

    def purgar_subidas(self, horas: float = HORAS_RETENCION_SUBIDAS) -> int:
        """Borra los videos subidos mas viejos que `horas`. Devuelve cuantos borro.

        Sin esto el directorio crece con cada peticion y no se vacia nunca. Se
        llama al recibir una subida nueva y no por un cron: el unico momento en
        que hace falta espacio es justo antes de ocuparlo.
        """
        if not self.dir_subidas.exists():
            return 0
        limite = time.time() - horas * 3600.0
        n = 0
        for f in self.dir_subidas.iterdir():
            try:
                if f.is_file() and f.stat().st_mtime < limite:
                    f.unlink()
                    n += 1
            except OSError:
                pass  # se lo llevo otro proceso, o no hay permiso: no es fatal
        return n

    def guardar_subida(self, nombre: str, flujo) -> Path:
        """Copia un video subido al directorio de subidas y devuelve su ruta.

        Se copia por bloques y con tope: `copyfileobj` sin limite deja que una
        sola peticion llene el disco del servidor, y ahi no falla solo el job
        —falla todo lo demas que necesite escribir, empezando por costs.db.
        """
        ext = Path(nombre or "").suffix.lower()
        if ext not in EXTENSIONES_VIDEO:
            raise FuenteInvalida(
                f"extension no soportada: '{ext or nombre}'. "
                f"Aceptadas: {', '.join(sorted(EXTENSIONES_VIDEO))}")
        self.dir_subidas.mkdir(parents=True, exist_ok=True)
        self.purgar_subidas()
        destino = self.dir_subidas / f"{uuid.uuid4().hex}{ext}"
        escrito = 0
        try:
            with open(destino, "wb") as f:
                while True:
                    bloque = flujo.read(1 << 20)
                    if not bloque:
                        break
                    escrito += len(bloque)
                    if escrito > MAX_SUBIDA_BYTES:
                        raise SubidaDemasiadoGrande(
                            f"el video supera el maximo de "
                            f"{MAX_SUBIDA_BYTES // (1024 * 1024)} MB")
                    f.write(bloque)
            if not escrito:
                raise FuenteInvalida("el archivo subido esta vacio")
        except Exception:
            destino.unlink(missing_ok=True)  # no dejar un archivo a medias
            raise
        return destino

    # ---------------------------------------------------------------- modelo

    def verificar_modelo(self, params: ParametrosModelo | None = None) -> tuple[bool, str]:
        """(disponible, detalle). Es lo que decide si se acepta un job.

        Se comprueba antes de aceptar y no al correr: un 503 en el POST es una
        respuesta; un job que arranca, se encola y falla a los diez segundos por
        una GPU que no existe es una molestia.
        """
        p = params or ParametrosModelo(backend=self.cfg.person.backend,
                                       device=self.cfg.person.device)
        try:
            import torch
            import ultralytics

            # Se toca el atributo a proposito: importar un paquete puede tener
            # exito y fallar despues al usarlo. Esto confirma que cargo de verdad.
            ultralytics.__version__
        except Exception as e:  # pragma: no cover - depende del entorno
            return False, f"no se pudo importar el stack de deteccion: {e}"

        if p.device != "cpu" and not torch.cuda.is_available():
            return False, (f"se pidio device='{p.device}' pero CUDA no esta "
                           "disponible en este proceso. Use device='cpu'.")

        from core.perception.detector import YOLO11_SEG, YOLO_WORLD, resolver_pesos

        nombre = YOLO11_SEG[0] if p.backend == "yolo11" else YOLO_WORLD[0]
        ruta = resolver_pesos(self.cfg.person.pesos or nombre)
        if Path(ruta).is_absolute() and not Path(ruta).exists():
            return False, f"faltan los pesos del backend '{p.backend}': {ruta}"
        return True, f"backend '{p.backend}' listo en device '{p.device}'"

    def _detector(self, cfg: CascadeConfig):
        """Detector cacheado por parametros. Levanta ModeloNoDisponible si no carga."""
        clave = (cfg.person.backend, cfg.person.pesos, cfg.person.imgsz,
                 cfg.person.conf, cfg.person.iou, cfg.person.device,
                 cfg.track.tracker)
        with self._lock_det:
            det = self._detectores.get(clave)
            if det is None:
                try:
                    det = construir_detector(cfg)
                    if hasattr(det, "calentar"):
                        # La primera inferencia paga la inicializacion de CUDA
                        # (44 ms medidos contra 10 de regimen). Si se paga aca, no
                        # se le carga al primer frame con movimiento del job.
                        det.calentar()
                except Exception as e:
                    raise ModeloNoDisponible(f"no se pudo cargar el detector: {e}") from e
                self._detectores[clave] = det
            return det

    # ---------------------------------------------------------------- config

    def config_de(self, peticion: PeticionProceso) -> CascadeConfig:
        """CascadeConfig del job: la del motor con los overrides de la peticion.

        Deriva de `self.cfg` y no de `load_cascade_config()` para que apuntar el
        motor a otro corpus o a otra db baste con construirlo distinto. `replace`
        devuelve copias: dos jobs con parametros distintos no se pisan.
        """
        return replace(
            self.cfg,
            person=replace(self.cfg.person, backend=peticion.modelo.backend,
                           conf=peticion.modelo.conf, iou=peticion.modelo.iou,
                           imgsz=peticion.modelo.imgsz,
                           device=peticion.modelo.device,
                           prompts=tuple(peticion.modelo.prompts)),
            conteo=replace(self.cfg.conteo, linea=tuple(peticion.conteo.linea)),
            reid=replace(self.cfg.reid, activo=peticion.usar_reid),
            filas=replace(self.cfg.filas, activo=peticion.roi is not None),
        )

    # ---------------------------------------------------------------- jobs

    def crear_job(self, peticion: PeticionProceso) -> RespuestaJob:
        """Valida, encola y devuelve el estado inicial. No espera a que termine.

        La respuesta se arma ANTES de encolar. Si se armara despues, un worker
        libre puede tomar el job entre el submit y la serializacion, y entonces
        el 202 contesta "running" unas veces y "queued" otras. Encolado es lo que
        es cierto en el instante en que se acepta.
        """
        self.resolver_fuente(peticion)  # 404 aca y no dentro del worker
        ok, detalle = self.verificar_modelo(peticion.modelo)
        if not ok:
            raise ModeloNoDisponible(detalle)
        job = self.almacen.crear(peticion)
        respuesta = job.a_respuesta()
        self._pool.submit(self._correr, job.job_id)
        return respuesta

    def _corte(self, job: Job, limite_s: float):
        """Closure que el bucle del runner consulta cada frame para saber si parar."""
        t0 = time.monotonic()

        def parar() -> bool:
            if job.cancelacion.is_set():
                return True
            return bool(limite_s) and (time.monotonic() - t0) >= limite_s

        return parar

    def _progreso(self, job: Job, t0: float):
        """Closure que el runner llama cada N frames para publicar avance."""

        def publicar(stats) -> None:
            self.almacen.actualizar(job.job_id, progreso=Progreso(
                frames_read=stats.total_frames,
                frames_processed=stats.frames_analizados,
                frames_discarded=stats.frames_descartados,
                people_so_far=stats.n_personas_unicas,
                elapsed_s=round(time.monotonic() - t0, 2),
            ))

        return publicar

    def _correr(self, job_id: str) -> None:
        """Cuerpo del worker. Corre en un hilo del pool, nunca en el de la peticion."""
        job = self.almacen.obtener(job_id)
        if job is None or job.terminal:
            return  # lo cancelaron mientras estaba en cola
        self.almacen.actualizar(job_id, estado=EstadoJob.RUNNING, iniciado=_ahora())

        cfg = self.config_de(job.peticion)
        # run_id = job_id: asi las mediciones de coste de este job se pueden leer
        # despues desde costs.db aunque el servicio se haya reiniciado.
        cost = CostTracker(cfg.costs_db, cpu_usd_per_hour=cfg.cpu_usd_per_hour,
                           gpu_usd_per_hour=cfg.gpu_usd_per_hour, run_id=job_id)
        t0 = time.monotonic()
        try:
            det = self._detector(cfg)
            r = job.peticion.roi
            roi = [tuple(p) for p in r.puntos] if r else None
            fila_cfg = FilaConfig(dwell_min_s=r.dwell_min_s, quieto_frac=r.quieto_frac,
                                  min_personas=r.min_personas) if r else None
            comun = dict(detector=det, max_frames=job.peticion.max_frames,
                         roi=roi, fila_cfg=fila_cfg,
                         debe_parar=self._corte(job, job.peticion.max_segundos),
                         progreso=self._progreso(job, t0))
            fuente = self.resolver_fuente(job.peticion)
            if job.peticion.fuente.tipo is TipoFuente.RTSP:
                stats = procesar_stream(fuente, cfg, cost, **comun)
            else:
                etiqueta = (job.peticion.fuente.valor
                            if job.peticion.fuente.tipo is TipoFuente.CLIP
                            else Path(fuente).stem)
                stats = procesar_clip(Path(fuente), etiqueta, cfg, cost,
                                      usar_gate=job.peticion.usar_filtro_movimiento,
                                      **comun)
            if stats.meta.get("error"):
                raise ErrorApi(stats.meta["error"])

            # Cancelado es solo lo que corto un DELETE. Agotar el `max_segundos`
            # que pidio el propio cliente es terminar bien: el runner marca
            # `stats.cancelado` en los dos casos porque para el son lo mismo
            # —una parada externa—, pero para el cliente no lo son.
            final = (EstadoJob.CANCELLED if job.cancelacion.is_set()
                     else EstadoJob.COMPLETED)
            self.almacen.actualizar(
                job_id, estado=final, stats=stats, terminado=_ahora(),
                progreso=Progreso(frames_read=stats.total_frames,
                                  frames_processed=stats.frames_analizados,
                                  frames_discarded=stats.frames_descartados,
                                  people_so_far=stats.n_personas_unicas,
                                  elapsed_s=round(time.monotonic() - t0, 2)))
        except Exception as e:
            # Se atrapa todo a proposito: si el worker muere con una excepcion, el
            # job queda para siempre en "running" y el cliente espera un final que
            # no llega. Mejor un "failed" con el motivo.
            self.almacen.actualizar(job_id, estado=EstadoJob.FAILED,
                                    error=f"{type(e).__name__}: {e}",
                                    terminado=_ahora())
        finally:
            cost.close()

    # ---------------------------------------------------------------- salidas

    def resultado_de(self, job: Job) -> ResultadoProceso:
        """Convierte las stats del runner en la respuesta acordada."""
        base = dict(job_id=job.job_id, status=job.estado,
                    source_type=job.peticion.fuente.tipo,
                    source=job.peticion.fuente.valor)
        s = job.stats
        if s is None:
            return ResultadoProceso(**base)

        cpu_ms = s.cpu_ms_motion + s.cpu_ms_tracking + s.cpu_ms_conteo
        gpu_ms = s.gpu_ms_person + s.gpu_ms_reid
        costo_usd = (cpu_ms / MS_POR_HORA * self.cfg.cpu_usd_per_hour
                     + gpu_ms / MS_POR_HORA * self.cfg.gpu_usd_per_hour)
        etapas = CostTracker(self.cfg.costs_db, run_id=job.job_id).resumen_por_stage()

        return ResultadoProceso(
            **base,
            people_count=s.n_personas_unicas,
            queue_detected=bool(s.hubo_fila),
            queue_size=s.max_en_fila,
            frames_processed=s.frames_analizados,
            frames_discarded=s.frames_descartados,
            processing_time=round(s.wall_ms / 1000.0, 3),
            estimated_cost=round(costo_usd, 8),
            detection=Deteccion(boxes=s.n_detecciones, mean_confidence=s.conf_media,
                                frames_with_people=s.frames_con_persona,
                                max_simultaneous=s.max_personas_simultaneas,
                                reid_reattachments=s.n_reenganches),
            counting=Conteo(entered=s.entradas, exited=s.salidas,
                            crossed_line=s.contados_por_linea),
            queue=Fila(roi_defined=s.tiene_roi, detected=bool(s.hubo_fila),
                       size=s.max_en_fila, max_in_zone=s.max_en_zona,
                       pct_frames_with_queue=s.pct_frames_con_fila),
            performance=Rendimiento(
                frames_total=s.total_frames, frames_processed=s.frames_analizados,
                frames_discarded=s.frames_descartados, pct_processed=s.pct_analizado,
                fps=s.fps, wall_time_s=round(s.wall_ms / 1000.0, 3),
                cpu_ms_motion=s.cpu_ms_motion, cpu_ms_tracking=s.cpu_ms_tracking,
                cpu_ms_counting=s.cpu_ms_conteo, gpu_ms_detection=s.gpu_ms_person,
                gpu_ms_reid=s.gpu_ms_reid),
            cost=Costo(cpu_ms_total=round(cpu_ms, 3), gpu_ms_total=round(gpu_ms, 3),
                       cpu_usd_per_hour=self.cfg.cpu_usd_per_hour,
                       gpu_usd_per_hour=self.cfg.gpu_usd_per_hour,
                       estimated_cost_usd=round(costo_usd, 8), per_stage=etapas),
        )

    def metricas_globales(self) -> MetricasGlobales:
        """Agregados de todos los jobs que este proceso conoce."""
        jobs = self.almacen.listar(limite=10_000)
        con_stats = [j.stats for j in jobs if j.stats is not None]
        total = sum(s.total_frames for s in con_stats)
        proc = sum(s.frames_analizados for s in con_stats)
        desc = sum(s.frames_descartados for s in con_stats)
        cpu = sum(s.cpu_ms_motion + s.cpu_ms_tracking + s.cpu_ms_conteo for s in con_stats)
        gpu = sum(s.gpu_ms_person + s.gpu_ms_reid for s in con_stats)
        utiles = proc + desc
        # Sin run_id: el resumen abarca TODAS las corridas escritas en la db, que
        # incluye los barridos por linea de comandos y no solo los jobs de la API.
        etapas = CostTracker(self.cfg.costs_db).resumen_por_stage(solo_esta_corrida=False)
        return MetricasGlobales(
            jobs_total=len(jobs), jobs_by_status=self.almacen.contar_por_estado(),
            frames_total=total, frames_processed=proc, frames_discarded=desc,
            pct_processed=round(100.0 * proc / utiles, 2) if utiles else 0.0,
            cpu_ms_total=round(cpu, 3), gpu_ms_total=round(gpu, 3),
            estimated_cost_usd=round(
                cpu / MS_POR_HORA * self.cfg.cpu_usd_per_hour
                + gpu / MS_POR_HORA * self.cfg.gpu_usd_per_hour, 8),
            per_stage=etapas)

    def salud(self) -> Salud:
        ok, detalle = self.verificar_modelo()
        cuentas = self.almacen.contar_por_estado()
        return Salud(status="ok" if ok else "degraded", model_available=ok,
                     model_detail=detalle, backend=self.cfg.person.backend,
                     device=self.cfg.person.device,
                     jobs_running=cuentas.get("running", 0),
                     jobs_queued=cuentas.get("queued", 0))

    def cerrar(self) -> None:
        """Para el pool. Los jobs en curso reciben la cancelacion primero."""
        for j in self.almacen.listar(limite=10_000):
            if not j.terminal:
                j.cancelacion.set()
        self._pool.shutdown(wait=False)
