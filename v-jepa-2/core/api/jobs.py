"""Registro de jobs en memoria, con acceso concurrente.

Vive en memoria a proposito y con la consecuencia escrita: **reiniciar el
servicio borra el estado de los jobs**. Lo que NO se pierde son las mediciones de
coste, que se siguen escribiendo en `results/costs.db` con el job_id como
`run_id`, asi que `GET /metricas` sobrevive al reinicio aunque `GET /jobs/{id}`
no. Persistir tambien el estado es cambiar `AlmacenJobs` por una tabla y nada
mas; hoy no hace falta.

El almacen tiene tope: sin el, un servicio de larga duracion acumula un job por
peticion y no libera nunca. Al pasarse, tira los terminados mas viejos y jamas
uno en curso.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.api.esquemas import (
    TERMINALES,
    EstadoJob,
    PeticionProceso,
    Progreso,
    RespuestaJob,
)

MAX_JOBS = 500  # tope del almacen; al pasarse se descartan terminados por antiguedad


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    """Un procesamiento y todo lo que se sabe de el."""

    job_id: str
    peticion: PeticionProceso
    estado: EstadoJob = EstadoJob.QUEUED
    creado: str = field(default_factory=_ahora)
    iniciado: str | None = None
    terminado: str | None = None
    error: str | None = None
    stats: object | None = None  # ClipPersonStats cuando termina bien
    progreso: Progreso = field(default_factory=Progreso)
    # Lo levanta DELETE /jobs/{id}; el bucle del runner lo consulta cada frame.
    cancelacion: threading.Event = field(default_factory=threading.Event)

    @property
    def terminal(self) -> bool:
        return self.estado in TERMINALES

    def a_respuesta(self) -> RespuestaJob:
        """Vista publica del job: es lo que devuelven POST /jobs y GET /jobs/{id}."""
        return RespuestaJob(
            job_id=self.job_id,
            status=self.estado,
            source_type=self.peticion.fuente.tipo,
            source=self.peticion.fuente.valor,
            created_at=self.creado,
            started_at=self.iniciado,
            finished_at=self.terminado,
            progress=self.progreso,
            error=self.error,
        )


class AlmacenJobs:
    """Diccionario job_id -> Job con un lock. Todo acceso pasa por aca.

    El worker escribe desde su propio hilo mientras el endpoint de estado lee
    desde el hilo de la peticion: sin el lock, un `GET` puede leer un job a medio
    actualizar.
    """

    def __init__(self, max_jobs: int = MAX_JOBS) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self.max_jobs = max_jobs

    def crear(self, peticion: PeticionProceso) -> Job:
        """Registra un job nuevo en estado queued y devuelve el objeto."""
        job = Job(job_id=uuid.uuid4().hex, peticion=peticion)
        with self._lock:
            self._jobs[job.job_id] = job
            self._podar()
        return job

    def _podar(self) -> None:
        """Descarta terminados viejos hasta volver al tope. Llamar con el lock tomado."""
        if len(self._jobs) <= self.max_jobs:
            return
        for jid in list(self._jobs):
            if len(self._jobs) <= self.max_jobs:
                break
            if self._jobs[jid].terminal:
                del self._jobs[jid]

    def obtener(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def listar(self, estado: EstadoJob | None = None, limite: int = 100) -> list[Job]:
        """Los mas nuevos primero, opcionalmente filtrados por estado."""
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.reverse()
        if estado is not None:
            jobs = [j for j in jobs if j.estado is estado]
        return jobs[:limite]

    def actualizar(self, job_id: str, **campos) -> Job | None:
        """Escribe campos del job bajo el lock. Devuelve el job, o None si no existe."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for k, v in campos.items():
                setattr(job, k, v)
            return job

    def cancelar(self, job_id: str) -> Job | None:
        """Pide la cancelacion. Un job en cola se corta ya; uno en curso, en el
        proximo frame. Uno ya terminado no se toca."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.terminal:
                return job
            job.cancelacion.set()
            if job.estado is EstadoJob.QUEUED:
                # Todavia no lo tomo el worker: se cierra aca mismo.
                job.estado = EstadoJob.CANCELLED
                job.terminado = _ahora()
            return job

    def contar_por_estado(self) -> dict[str, int]:
        with self._lock:
            jobs = list(self._jobs.values())
        cuentas: dict[str, int] = {}
        for j in jobs:
            cuentas[j.estado.value] = cuentas.get(j.estado.value, 0) + 1
        return cuentas
