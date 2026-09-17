"""Detector open-vocabulary: arranca la deteccion desde prompts en lenguaje natural.

Permite que la camara empiece con una "idea basica" expresada en texto
("una persona", "alguien con un carrito") sin entrenamiento previo de clases.

Backends:
- YOLO-World      (arxiv 2401.17270) -- open-vocab por texto, el de yolo_world/
- YOLO11-seg      -- clases COCO fijas, mas rapido, el de yolo_seg/
- Grounding DINO  (IDEA-Research, arxiv 2303.05499) -- pendiente

Los dos primeros son los mismos modelos y los mismos parametros que usan los
notebooks de `yolo_world/` y `yolo_seg/`; esto es el adapter importable de lo que
ahi se corre a mano en Colab, para que el cascade y el orquestador compartan
detector en vez de tener cada uno el suyo.

El seguimiento NO es una etapa aparte: ultralytics corre ByteTrack adentro del
propio predictor (`model.track(persist=True)`), asi que el track_id llega junto
con la caja. Ver core/perception/tracker.py.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from core.types import Detection

# Los mismos que declaran los notebooks, para que los resultados sean comparables.
YOLO11_SEG = ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt"]
YOLO_WORLD = ["yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt"]
TRACKER_POR_DEFECTO = "bytetrack.yaml"  # alternativa: botsort.yaml

# En el contenedor los pesos vienen dentro de la imagen. Fuera (Colab, local) la
# variable no existe y ultralytics los baja solo, como en los notebooks.
MODELS_DIR = os.environ.get("AMTA_MODELS_DIR", "")


def resolver_pesos(nombre: str) -> str:
    """Path absoluto a los pesos si estan en la imagen; si no, el nombre pelado.

    Con el nombre pelado ultralytics los descarga al directorio actual, que es lo
    que pasa en Colab. En el contenedor eso ademas fallaria: corre como no-root y
    no puede escribir en /app.
    """
    if MODELS_DIR:
        p = Path(MODELS_DIR) / nombre
        if p.exists():
            return str(p)
    return nombre


class OpenVocabDetector(ABC):
    """Interfaz comun para detectores guiados por prompts de texto."""

    @abstractmethod
    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Detecta en `frame` (HxWx3 RGB) los conceptos descritos en `prompts`."""
        raise NotImplementedError

    def track(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Como detect(), pero las Detection vuelven con track_id de ByteTrack.

        Por defecto no hay seguimiento; los backends de ultralytics lo sobreescriben.
        """
        return self.detect(frame, prompts)


class _UltralyticsDetector(OpenVocabDetector):
    """Base para los backends de ultralytics. No usar directo.

    Concentra lo unico que de verdad comparten YOLO11 y YOLO-World: la conversion
    del `Results` de ultralytics a la lista de `Detection` del proyecto, y la
    medicion del tiempo de GPU.
    """

    def __init__(self, pesos: str, imgsz: int = 640, conf: float = 0.35,
                 iou: float = 0.70, device: str = "0",
                 tracker: str = TRACKER_POR_DEFECTO) -> None:
        self.pesos = pesos
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.tracker = tracker
        self.gpu_time_ms = 0.0  # tiempo de la ultima llamada, lo lee el medidor
        self._modelo = None

    def _cargar(self):
        raise NotImplementedError

    @property
    def modelo(self):
        """Carga perezosa: importar ultralytics arrastra torch y tarda segundos."""
        if self._modelo is None:
            self._modelo = self._cargar()
        return self._modelo

    def _sincronizar(self) -> None:
        """Espera a que la GPU termine, para que el tiempo medido sea real.

        Sin esto las llamadas CUDA son asincronas y el reloj mide el encolado, no
        el computo: el Nivel 2 pareceria costar casi cero.
        """
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _predecir(self, frame: np.ndarray, prompts: list[str], seguir: bool):
        raise NotImplementedError

    def _ejecutar(self, frame: np.ndarray, prompts: list[str],
                  seguir: bool) -> list[Detection]:
        self._sincronizar()
        t0 = time.perf_counter_ns()
        resultado = self._predecir(frame, prompts, seguir)
        self._sincronizar()
        self.gpu_time_ms = (time.perf_counter_ns() - t0) / 1e6
        return self._a_detections(resultado)

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        return self._ejecutar(frame, prompts, seguir=False)

    def track(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        return self._ejecutar(frame, prompts, seguir=True)

    def calentar(self, alto: int = 480, ancho: int = 640) -> float:
        """Corre una inferencia de descarte y devuelve cuanto tardo, en ms.

        La primera llamada a CUDA paga la inicializacion del contexto y la
        seleccion de kernels: medido en esta maquina, 44 ms/frame contra los 10
        de regimen. Sin calentar, ese costo cae entero sobre el primer clip que
        tenga movimiento y el C2 del reporte sale inflado.
        """
        import numpy as np

        self.detect(np.zeros((alto, ancho, 3), dtype=np.uint8), ["person"])
        return self.gpu_time_ms

    def reset(self) -> None:
        """Olvida el estado de ByteTrack. Obligatorio entre clips distintos.

        `persist=True` mantiene los tracks entre llamadas, que es lo que se quiere
        dentro de un clip y lo que NO se quiere al pasar al siguiente: sin esto la
        primera persona del clip nuevo hereda el ID de la ultima del anterior.
        """
        if self._modelo is not None and getattr(self._modelo, "predictor", None):
            if getattr(self._modelo.predictor, "trackers", None):
                for t in self._modelo.predictor.trackers:
                    t.reset()

    def _a_detections(self, resultado) -> list[Detection]:
        """Convierte un Results de ultralytics a la lista de Detection del proyecto.

        Si el modelo es `-seg` tambien se leen las mascaras. Ya vienen calculadas
        —el forward es el mismo— y hasta ahora se tiraban. Sirven para recortar a
        la persona sin fondo, que es lo que necesita el Nivel 3: en una tienda dos
        personas distintas delante de la misma gondola comparten medio recorte y
        sus embeddings se parecen por el fondo, no por ellas.
        """
        cajas = getattr(resultado, "boxes", None)
        if cajas is None or len(cajas) == 0:
            return []
        xyxy = cajas.xyxy.cpu().numpy()
        conf = cajas.conf.cpu().numpy()
        clases = cajas.cls.cpu().numpy().astype(int)
        ids = (cajas.id.cpu().numpy().astype(int)
               if getattr(cajas, "id", None) is not None else None)
        nombres = resultado.names
        m = getattr(resultado, "masks", None)
        mascaras = m.data.cpu().numpy() if m is not None and m.data is not None else None

        salida = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            salida.append(Detection(
                bbox=(x1, y1, x2, y2),
                label=str(nombres.get(clases[i], clases[i])),
                score=float(conf[i]),
                track_id=int(ids[i]) if ids is not None else None,
                mask=mascaras[i] if mascaras is not None and i < len(mascaras) else None,
            ))
        return salida


class Yolo11Detector(_UltralyticsDetector):
    """YOLO11-seg con clases COCO fijas. El backend de `yolo_seg/`.

    No es open-vocab: los prompts se traducen a los indices de clase de COCO que
    coincidan por nombre. Si ninguno coincide, no filtra por clase.
    """

    def __init__(self, pesos: str = YOLO11_SEG[0], **kw) -> None:
        super().__init__(resolver_pesos(pesos), **kw)

    def _cargar(self):
        from ultralytics import YOLO

        return YOLO(self.pesos)

    def _clases_de(self, prompts: list[str]) -> list[int] | None:
        nombres = {v: k for k, v in self.modelo.names.items()}
        idx = [nombres[p] for p in prompts if p in nombres]
        return idx or None

    def _predecir(self, frame, prompts, seguir):
        comun = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                     device=self.device, classes=self._clases_de(prompts),
                     verbose=False)
        if seguir:
            return self.modelo.track(frame, persist=True, tracker=self.tracker,
                                     **comun)[0]
        return self.modelo.predict(frame, **comun)[0]


class YoloWorldDetector(_UltralyticsDetector):
    """YOLO-World: deteccion open-vocab por texto. El backend de `yolo_world/`."""

    def __init__(self, pesos: str = YOLO_WORLD[0], conf: float = 0.25, **kw) -> None:
        super().__init__(resolver_pesos(pesos), conf=conf, **kw)
        self._prompts_puestos: list[str] | None = None

    def _cargar(self):
        from ultralytics import YOLOWorld

        return YOLOWorld(self.pesos)

    def _predecir(self, frame, prompts, seguir):
        # set_classes reconstruye los embeddings de texto: es caro, solo si cambio.
        if prompts != self._prompts_puestos:
            self.modelo.set_classes(prompts)
            self._prompts_puestos = list(prompts)
        comun = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                     device=self.device, verbose=False)
        if seguir:
            return self.modelo.track(frame, persist=True, tracker=self.tracker,
                                     **comun)[0]
        return self.modelo.predict(frame, **comun)[0]


class GroundingDinoDetector(OpenVocabDetector):
    """Adapter para Grounding DINO. TODO: cargar pesos y mapear salida a Detection."""

    def __init__(self, weights: str | None = None, box_threshold: float = 0.3):
        self.weights = weights
        self.box_threshold = box_threshold

    def detect(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        raise NotImplementedError("Pendiente: integrar Grounding DINO")


BACKENDS = {"yolo11": Yolo11Detector, "yoloworld": YoloWorldDetector}


def crear_detector(backend: str, **kw) -> OpenVocabDetector:
    """Fabrica por nombre. `backend` es 'yolo11' o 'yoloworld'."""
    if backend not in BACKENDS:
        raise ValueError(f"backend desconocido: {backend}. Opciones: {sorted(BACKENDS)}")
    return BACKENDS[backend](**kw)
