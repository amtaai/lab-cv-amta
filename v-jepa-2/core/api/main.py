"""API REST del cascade: endpoints, codigos de error y documentacion.

Corre con:
    uvicorn core.api.main:app --host 0.0.0.0 --port 8000
La documentacion interactiva queda en /docs (la genera FastAPI desde los
esquemas de `esquemas.py`, asi que nunca se desincroniza del codigo).

Este modulo no importa nada de `core.cascade`: todo el trabajo pasa por `Motor`.
Es lo que hace que los tests puedan correr la API entera sin GPU ni pesos,
sustituyendo el motor por uno falso.

Mapa de codigos:
  202  el job se acepto y quedo encolado; procesar lleva tiempo y no se espera
  200  consulta resuelta
  400  la fuente es inutilizable (ruta fuera del area servida, archivo vacio)
  413  el video subido supera AMTA_API_MAX_SUBIDA_MB (2048 por defecto)
  404  el job, el clip o el archivo no existen
  409  se pidieron resultados de un job que todavia no termino
  422  la peticion no valida contra el esquema (lo genera Pydantic)
  503  el detector no esta disponible (faltan pesos, o se pidio GPU y no hay)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Path, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from core.api.esquemas import (
    ErrorRespuesta,
    EstadoJob,
    IdJob,
    MetricasGlobales,
    PeticionProceso,
    ResultadoProceso,
    RespuestaJob,
    Salud,
    TipoFuente,
)
from core.api.servicio import ErrorApi, Motor

_motor: Motor | None = None


def get_motor() -> Motor:
    """Dependencia del motor. Los tests la sobreescriben con app.dependency_overrides."""
    global _motor
    if _motor is None:
        _motor = Motor()
    return _motor


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_motor()
    yield
    if _motor is not None:
        _motor.cerrar()


app = FastAPI(
    title="Cascade de vigilancia — API",
    version="1.0.0",
    summary="Deteccion de movimiento, personas, seguimiento, conteo y filas sobre "
            "video o stream RTSP.",
    description=__doc__,
    lifespan=lifespan,
)

RESPUESTAS_JOB = {
    404: {"model": ErrorRespuesta, "description": "no existe ese job"},
}


@app.exception_handler(ErrorApi)
async def _manejar_error_api(request, exc: ErrorApi):
    """Un solo lugar donde los errores del motor se vuelven respuestas HTTP."""
    return JSONResponse(status_code=exc.http,
                        content={"detail": exc.detalle, "codigo": exc.codigo})


def _buscar(motor: Motor, job_id: str):
    job = motor.almacen.obtener(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no existe el job '{job_id}'")
    return job


ParamJobId = Annotated[IdJob, Path(description="identificador devuelto por POST /jobs")]
MotorDep = Annotated[Motor, Depends(get_motor)]


@app.get("/salud", response_model=Salud, tags=["servicio"],
         summary="Estado del servicio y disponibilidad del modelo")
def salud(motor: MotorDep) -> Salud:
    """`degraded` significa que el servicio responde pero rechazaria jobs con 503."""
    return motor.salud()


@app.post("/jobs", response_model=RespuestaJob, status_code=202, tags=["procesamiento"],
          summary="Iniciar el procesamiento de un video o un stream",
          responses={
              400: {"model": ErrorRespuesta, "description": "fuente inutilizable"},
              404: {"model": ErrorRespuesta, "description": "el clip o archivo no existe"},
              503: {"model": ErrorRespuesta, "description": "modelo no disponible"},
          })
def crear_job(peticion: PeticionProceso, motor: MotorDep) -> RespuestaJob:
    """Encola el procesamiento y responde 202 sin esperarlo.

    Un video de un minuto tarda decenas de segundos: devolver el resultado en la
    misma peticion daria timeouts en cualquier proxy. El cliente consulta despues
    `GET /jobs/{job_id}` hasta que `status` sea terminal.
    """
    return motor.crear_job(peticion)


@app.post("/jobs/video", response_model=RespuestaJob, status_code=202,
          tags=["procesamiento"], summary="Subir un video y procesarlo",
          responses={
              400: {"model": ErrorRespuesta, "description": "archivo invalido o vacio"},
              413: {"model": ErrorRespuesta, "description": "el video supera el tope"},
              503: {"model": ErrorRespuesta, "description": "modelo no disponible"},
          })
def subir_y_procesar(
    motor: MotorDep,
    archivo: Annotated[UploadFile, File(description="video a procesar")],
    peticion: Annotated[Optional[str], Form(
        description="PeticionProceso en JSON, sin el campo 'fuente'. "
                    "Si se omite, se usan los valores por defecto.")] = None,
) -> RespuestaJob:
    """Igual que POST /jobs pero el video viaja en la peticion.

    Los parametros van en un campo de formulario con el mismo JSON que acepta
    POST /jobs, para no tener dos esquemas de entrada que se desincronicen.
    """
    datos = {}
    if peticion:
        try:
            datos = json.loads(peticion)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=422,
                                detail=f"el campo 'peticion' no es JSON valido: {e}")
        if not isinstance(datos, dict):
            raise HTTPException(status_code=422,
                                detail="el campo 'peticion' tiene que ser un objeto JSON")
    ruta = motor.guardar_subida(archivo.filename or "", archivo.file)
    datos["fuente"] = {"tipo": TipoFuente.ARCHIVO.value, "valor": str(ruta)}
    try:
        completa = PeticionProceso.model_validate(datos)
    except ValidationError as e:
        ruta.unlink(missing_ok=True)  # no dejar el archivo si la peticion no vale
        raise HTTPException(status_code=422, detail=json.loads(e.json()))
    return motor.crear_job(completa)


@app.get("/jobs", response_model=list[RespuestaJob], tags=["procesamiento"],
         summary="Listar procesamientos, los mas nuevos primero")
def listar_jobs(
    motor: MotorDep,
    status: Annotated[Optional[EstadoJob], Query(description="filtrar por estado")] = None,
    limite: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[RespuestaJob]:
    return [j.a_respuesta() for j in motor.almacen.listar(estado=status, limite=limite)]


@app.get("/jobs/{job_id}", response_model=RespuestaJob, tags=["procesamiento"],
         summary="Consultar el estado de un procesamiento", responses=RESPUESTAS_JOB)
def estado_job(job_id: ParamJobId, motor: MotorDep) -> RespuestaJob:
    """Mientras `status` es `running`, `progress` avanza; en terminal, ya no cambia."""
    return _buscar(motor, job_id).a_respuesta()


@app.delete("/jobs/{job_id}", response_model=RespuestaJob, tags=["procesamiento"],
            summary="Cancelar un procesamiento", responses=RESPUESTAS_JOB)
def cancelar_job(job_id: ParamJobId, motor: MotorDep) -> RespuestaJob:
    """Un job en cola se cancela al instante; uno en curso, en el proximo frame.

    Es la unica forma de terminar un job sobre RTSP sin tope: un stream en vivo
    no se acaba solo.
    """
    _buscar(motor, job_id)
    return motor.almacen.cancelar(job_id).a_respuesta()


@app.get("/jobs/{job_id}/resultados", response_model=ResultadoProceso,
         tags=["resultados"], summary="Obtener los resultados de un procesamiento",
         responses={**RESPUESTAS_JOB,
                    409: {"model": ErrorRespuesta,
                          "description": "el job todavia no termino"}})
def resultados_job(job_id: ParamJobId, motor: MotorDep) -> ResultadoProceso:
    """409 mientras el job no sea terminal.

    Devolver resultados parciales con el mismo esquema que los finales invita a
    tomar un conteo a medias por definitivo. El avance se mira en `GET /jobs/{id}`.
    """
    job = _buscar(motor, job_id)
    if not job.terminal:
        raise HTTPException(
            status_code=409,
            detail=f"el job '{job_id}' esta en estado '{job.estado.value}'. "
                   f"Consulte GET /jobs/{job_id} hasta que termine.")
    return motor.resultado_de(job)


@app.get("/jobs/{job_id}/metricas", tags=["metricas"],
         summary="Rendimiento y coste de un procesamiento", responses=RESPUESTAS_JOB)
def metricas_job(job_id: ParamJobId, motor: MotorDep) -> dict:
    """Solo la parte de rendimiento y coste, para monitoreo.

    Se sirve tambien de un job cancelado o fallido: lo que se alcanzo a procesar
    se pago igual y tiene que poder contabilizarse.
    """
    job = _buscar(motor, job_id)
    r = motor.resultado_de(job)
    return {"job_id": job.job_id, "status": job.estado,
            "performance": r.performance, "cost": r.cost}


@app.get("/metricas", response_model=MetricasGlobales, tags=["metricas"],
         summary="Rendimiento y coste agregados del servicio")
def metricas(motor: MotorDep) -> MetricasGlobales:
    """`per_stage` sale del log de coste en SQLite y abarca TODAS las corridas
    escritas ahi, incluidos los barridos por linea de comandos. El resto de los
    campos son solo los jobs que este proceso conoce."""
    return motor.metricas_globales()
