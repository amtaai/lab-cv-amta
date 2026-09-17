"""Esquemas Pydantic de entrada y salida de la API.

Los nombres de los campos de SALIDA estan en ingles y los de ENTRADA en espanol.
No es un descuido: la forma de la respuesta es un contrato acordado
(`people_count`, `queue_detected`, `frames_discarded`, ...) y se respeta tal cual;
la peticion es nuestra y sigue la convencion del resto del repo. Adentro todo
sigue siendo espanol, como el resto de `core/`.

La validacion es la razon de ser de este modulo. Todo lo que aca se declara con
un rango es algo que, mal puesto, hace que el pipeline falle tarde y mal:
una `conf` de 1.5 no detecta nada y no avisa, una ROI con dos puntos revienta
adentro de shapely, un `imgsz` que no es multiplo de 32 hace que ultralytics
reescale por su cuenta y los ms/frame medidos dejen de ser comparables.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Coordenada normalizada [0,1]. TODA la geometria de la API es normalizada, nunca
# en pixeles: el corpus mezcla 640x480 con 1920x1080 y una linea en pixeles cae
# en lugares distintos en cada resolucion. Es el mismo criterio que ya usan
# ConteoConfig y corpus/rois.json.
Punto = tuple[
    Annotated[float, Field(ge=0.0, le=1.0)],
    Annotated[float, Field(ge=0.0, le=1.0)],
]

# job_id: se genera como uuid4().hex, pero se acepta cualquier token razonable.
# Asi un id desconocido da 404 (que es lo que significa) y solo un id imposible
# de haber emitido nunca da 422.
IdJob = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]


class TipoFuente(str, Enum):
    """De donde sale el video."""

    CLIP = "clip"  # clip_id ya ingestado en corpus/metadata.json
    ARCHIVO = "archivo"  # ruta a un archivo de video accesible por el servicio
    RTSP = "rtsp"  # stream en vivo


class EstadoJob(str, Enum):
    """Ciclo de vida de un procesamiento."""

    QUEUED = "queued"  # aceptado, esperando turno (la GPU se usa de a uno)
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"  # lo corto un DELETE, no un error


TERMINALES = {EstadoJob.COMPLETED, EstadoJob.FAILED, EstadoJob.CANCELLED}


class Fuente(BaseModel):
    """Que procesar. El `valor` se valida segun el `tipo`."""

    model_config = ConfigDict(extra="forbid")

    tipo: TipoFuente
    valor: str = Field(min_length=1, max_length=1024,
                       description="clip_id, ruta del archivo o URL rtsp://")

    @model_validator(mode="after")
    def _valor_coherente_con_tipo(self) -> Fuente:
        v = self.valor.strip()
        if not v:
            raise ValueError("valor vacio")
        if self.tipo is TipoFuente.RTSP:
            if not v.startswith(("rtsp://", "rtsps://")):
                raise ValueError("una fuente rtsp tiene que empezar con rtsp:// o rtsps://")
        elif self.tipo is TipoFuente.CLIP:
            # El clip_id indexa corpus/raw/: si dejara pasar separadores, un
            # clip_id como "../../etc/passwd" saldria del corpus.
            if "/" in v or "\\" in v or v.startswith("."):
                raise ValueError("clip_id invalido: sin separadores de ruta ni punto inicial")
        return self


class ParametrosModelo(BaseModel):
    """Parametros del Nivel 2. Los defaults son los de CascadeConfig."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["yolo11", "yoloworld"] = "yolo11"
    conf: float = Field(0.35, ge=0.0, le=1.0, description="umbral de confianza")
    iou: float = Field(0.70, ge=0.0, le=1.0, description="IoU del NMS")
    imgsz: int = Field(640, ge=320, le=1920, description="lado de entrada, multiplo de 32")
    device: str = Field("0", pattern=r"^(cpu|cuda:\d+|\d+)$")
    prompts: list[str] = Field(default_factory=lambda: ["person"], min_length=1, max_length=16)

    @field_validator("imgsz")
    @classmethod
    def _multiplo_de_32(cls, v: int) -> int:
        # Si no es multiplo de 32 ultralytics reescala por su cuenta y en silencio:
        # los ms/frame dejan de ser comparables con los ya medidos.
        if v % 32:
            raise ValueError("imgsz tiene que ser multiplo de 32")
        return v

    @field_validator("prompts")
    @classmethod
    def _prompts_no_vacios(cls, v: list[str]) -> list[str]:
        limpios = [p.strip() for p in v if p.strip()]
        if not limpios:
            raise ValueError("prompts no puede quedar vacio")
        return limpios


class ParametrosROI(BaseModel):
    """Zona de espera para la deteccion de filas, en coordenadas normalizadas."""

    model_config = ConfigDict(extra="forbid")

    puntos: list[Punto] = Field(min_length=3, max_length=32,
                                description="poligono, minimo un triangulo")
    dwell_min_s: float = Field(2.0, ge=0.0, le=600.0,
                               description="segundos quieto para contar como esperando")
    quieto_frac: float = Field(0.06, ge=0.0, le=1.0,
                               description="desplazamiento maximo, como fraccion del frame")
    min_personas: int = Field(2, ge=1, le=100, description="desde cuantos hay fila")

    @field_validator("puntos")
    @classmethod
    def _sin_puntos_repetidos(cls, v: list) -> list:
        # Un poligono con vertices duplicados degenera y shapely devuelve area 0:
        # la ROI existiria pero nadie caeria adentro nunca.
        if len({tuple(p) for p in v}) < 3:
            raise ValueError("la ROI necesita al menos 3 vertices distintos")
        return v


class ParametrosConteo(BaseModel):
    """Linea de conteo: quien la cruza en un sentido entra, en el otro sale."""

    model_config = ConfigDict(extra="forbid")

    linea: tuple[Punto, Punto] = ((0.35, 0.0), (0.35, 1.0))

    @field_validator("linea")
    @classmethod
    def _linea_no_degenerada(cls, v):
        if tuple(v[0]) == tuple(v[1]):
            raise ValueError("los dos extremos de la linea no pueden coincidir")
        return v


class PeticionProceso(BaseModel):
    """Cuerpo de POST /jobs."""

    model_config = ConfigDict(extra="forbid")

    fuente: Fuente
    modelo: ParametrosModelo = Field(default_factory=ParametrosModelo)
    conteo: ParametrosConteo = Field(default_factory=ParametrosConteo)
    roi: Optional[ParametrosROI] = None
    usar_filtro_movimiento: bool = Field(
        True, description="False saltea el Nivel 1: es la corrida de control del coste")
    usar_reid: bool = Field(True, description="Nivel 3: reengancha IDs partidos por oclusion")
    max_frames: int = Field(0, ge=0, le=5_000_000, description="0 = sin tope")
    max_segundos: float = Field(0.0, ge=0.0, le=86_400.0, description="0 = sin tope")

    @model_validator(mode="after")
    def _stream_necesita_tope(self) -> PeticionProceso:
        # Un RTSP no termina nunca. Sin tope el job solo se corta con DELETE, lo
        # cual es legitimo, pero hay que decirlo aca y no descubrirlo despues.
        if (self.fuente.tipo is TipoFuente.RTSP
                and not self.max_frames and not self.max_segundos):
            self.max_segundos = 60.0
        return self


# --------------------------------------------------------------------------
# Salida
# --------------------------------------------------------------------------

class Progreso(BaseModel):
    """Avance de un job en curso. Cero mientras esta en cola."""

    frames_read: int = 0
    frames_processed: int = 0
    frames_discarded: int = 0
    people_so_far: int = 0
    elapsed_s: float = 0.0


class RespuestaJob(BaseModel):
    """Estado de un procesamiento. Es lo que devuelve POST /jobs y GET /jobs/{id}."""

    job_id: str
    status: EstadoJob
    source_type: TipoFuente
    source: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: Progreso = Field(default_factory=Progreso)
    error: Optional[str] = None


class Deteccion(BaseModel):
    """Lo que vio el Nivel 2."""

    boxes: int = 0  # cajas totales: una persona en 50 frames son 50 cajas
    mean_confidence: float = 0.0
    frames_with_people: int = 0
    max_simultaneous: int = 0
    reid_reattachments: int = 0


class Conteo(BaseModel):
    """Punto 3: quien cruzo la linea y en que sentido."""

    entered: int = 0
    exited: int = 0
    crossed_line: int = 0


class Fila(BaseModel):
    """Punto 4: fila en la zona de espera. Solo tiene sentido si hay ROI."""

    roi_defined: bool = False
    detected: bool = False
    size: int = 0  # maximo de personas esperando a la vez
    max_in_zone: int = 0
    pct_frames_with_queue: float = 0.0


class Rendimiento(BaseModel):
    """Punto 7: cuanto trabajo se hizo y a que ritmo.

    CPU y GPU van separadas a proposito: son recursos distintos y sumarlas
    esconde cual de los dos hay que pagar.
    """

    frames_total: int = 0
    frames_processed: int = 0  # los que el Nivel 1 dejo pasar al detector
    frames_discarded: int = 0  # el ahorro del filtro
    pct_processed: float = 0.0
    fps: float = 0.0
    wall_time_s: float = 0.0
    cpu_ms_motion: float = 0.0
    cpu_ms_tracking: float = 0.0
    cpu_ms_counting: float = 0.0
    gpu_ms_detection: float = 0.0
    gpu_ms_reid: float = 0.0


class Costo(BaseModel):
    """Coste del job. En dolares es 0 mientras las tarifas esten en 0, que es el default."""

    cpu_ms_total: float = 0.0
    gpu_ms_total: float = 0.0
    cpu_usd_per_hour: float = 0.0
    gpu_usd_per_hour: float = 0.0
    estimated_cost_usd: float = 0.0
    per_stage: dict = Field(default_factory=dict)


class ResultadoProceso(BaseModel):
    """Cuerpo de GET /jobs/{id}/resultados.

    Los ocho campos de arriba son el contrato acordado y estan planos a
    proposito; lo demas cuelga anidado para no romperlo al agregar cosas.
    """

    job_id: str
    status: EstadoJob
    people_count: int = 0
    queue_detected: bool = False
    queue_size: int = 0
    frames_processed: int = 0
    frames_discarded: int = 0
    processing_time: float = 0.0
    estimated_cost: float = 0.0

    source_type: TipoFuente
    source: str
    detection: Deteccion = Field(default_factory=Deteccion)
    counting: Conteo = Field(default_factory=Conteo)
    queue: Fila = Field(default_factory=Fila)
    performance: Rendimiento = Field(default_factory=Rendimiento)
    cost: Costo = Field(default_factory=Costo)


class MetricasGlobales(BaseModel):
    """Cuerpo de GET /metricas: agregados de todas las corridas del log de coste."""

    jobs_total: int = 0
    jobs_by_status: dict = Field(default_factory=dict)
    frames_total: int = 0
    frames_processed: int = 0
    frames_discarded: int = 0
    pct_processed: float = 0.0
    cpu_ms_total: float = 0.0
    gpu_ms_total: float = 0.0
    estimated_cost_usd: float = 0.0
    per_stage: dict = Field(default_factory=dict)


class Salud(BaseModel):
    """Cuerpo de GET /salud. `model_available` es lo que decide si se aceptan jobs."""

    status: Literal["ok", "degraded"]
    model_available: bool
    model_detail: str
    backend: str
    device: str
    jobs_running: int = 0
    jobs_queued: int = 0


class ErrorRespuesta(BaseModel):
    """Cuerpo de cualquier error de la API. Una sola forma para todos los codigos."""

    detail: str
    codigo: str = ""
