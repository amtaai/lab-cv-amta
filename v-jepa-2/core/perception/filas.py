"""Deteccion de filas sobre una zona de espera (ROI).

Se apoya en `ultralytics.solutions.QueueManager` para el conteo dentro de la
region, con el mismo mixin de tracks externos que usa el contador por linea: la
clase de ultralytics corre su propio `model.track()` y hay que sobreescribir
`extract_tracks` para que consuma los tracks que el cascade ya calculo.

LO QUE ULTRALYTICS NO HACE. `QueueManager` cuenta a cualquiera que este dentro de
la region, camine o no. Eso no es una fila: por un pasillo pasa gente que no
espera nada. Para contar como "en fila" se le pide ademas:

- permanencia: llevar al menos `dwell_min_s` segundos dentro de la ROI;
- quietud: haberse desplazado menos de `quieto_frac` del ancho del frame en ese
  rato. El que cruza caminando se mueve mucho mas.

Y hay fila recien con `min_personas`: una persona parada no es una fila, es
alguien siendo atendido.

POR QUE LA ROI ES POR CLIP. Una version anterior de esto se quito del proyecto
porque usaba UNA ROI global: cada camara tiene su geometria, y un rectangulo que
sirve para una cae en cualquier lado en las demas. El numero que salia no
describia nada. Ahora la ROI se declara por clip en corpus/rois.json y un clip
sin ROI declarada simplemente no se evalua, en vez de evaluarse mal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from core.perception.conteo import _MixinTracksExternos, _kwargs_base, _preparar, utiles
from core.types import Detection


@dataclass
class FilaConfig:
    """Que cuenta como fila."""

    # Cuanto hay que quedarse en la ROI para contar como esperando y no como
    # pasando. Debajo de esto, cualquiera que cruce la zona contaria de fila.
    dwell_min_s: float = 2.0
    # Desplazamiento maximo durante esa espera, como fraccion del ancho del frame.
    quieto_frac: float = 0.06
    # Una persona parada no es una fila: es alguien siendo atendido.
    min_personas: int = 2


@dataclass
class EstadoFila:
    """Que vio la zona en un frame."""

    en_zona: int = 0   # personas dentro de la ROI ahora
    en_fila: int = 0   # de esas, cuantas estan realmente esperando
    hay_fila: bool = False
    meta: dict = field(default_factory=dict)


def cargar_rois(path) -> dict:
    """Lee corpus/rois.json: clip_id -> poligono normalizado. Vacio si no existe."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: [tuple(p) for p in v["roi"]] for k, v in d.get("clips", {}).items()}


def centro(det: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = det.bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class DetectorFila:
    """Cuenta quien espera dentro de la ROI, y decide si eso es una fila."""

    def __init__(self, roi_norm, ancho: int, alto: int, fps: float = 25.0,
                 cfg: FilaConfig | None = None, line_width: int = 2):
        from shapely.geometry import Polygon

        from ultralytics.solutions import QueueManager

        class _Cola(_MixinTracksExternos, QueueManager):
            pass

        from core.perception.conteo import a_pixeles

        self.cfg = cfg or FilaConfig()
        self.roi_px = a_pixeles(roi_norm, ancho, alto)
        self._poligono = Polygon(self.roi_px)
        self._sol = _Cola(**_kwargs_base(self.roi_px, line_width))
        _preparar(self._sol, self.roi_px)
        self.fps = max(1.0, fps)
        self.dwell_min_frames = max(1, int(round(self.cfg.dwell_min_s * self.fps)))
        self.quieto_px = self.cfg.quieto_frac * ancho
        self.max_en_zona = 0
        self.max_en_fila = 0
        self.frames_con_fila = 0
        self.frames_evaluados = 0
        # track_id -> centroides mientras estuvo dentro de la ROI
        self._dentro: dict[int, list] = {}

    def _adentro(self, det: Detection) -> bool:
        from shapely.geometry import Point

        return self._poligono.contains(Point(*centro(det)))

    def actualizar(self, frame: np.ndarray, dets: list[Detection]) -> EstadoFila:
        """Procesa un frame y devuelve el estado de la zona."""
        confirmados = utiles(dets)
        self._sol.cargar(confirmados)
        self._sol.process_queue(frame)
        en_zona = int(self._sol.counts)  # el numero que da ultralytics
        self.frames_evaluados += 1

        vistos, en_fila = set(), 0
        for d in confirmados:
            if not self._adentro(d):
                continue
            tid = int(d.track_id)
            vistos.add(tid)
            hist = self._dentro.setdefault(tid, [])
            hist.append(centro(d))
            if len(hist) < self.dwell_min_frames:
                continue
            # Solo la ventana de permanencia minima: mirando toda la historia,
            # alguien que llego caminando nunca bajaria del umbral de quietud.
            ventana = hist[-self.dwell_min_frames:]
            xs = [p[0] for p in ventana]
            ys = [p[1] for p in ventana]
            if max(max(xs) - min(xs), max(ys) - min(ys)) <= self.quieto_px:
                en_fila += 1

        # El que salio de la ROI pierde su permanencia: si vuelve, vuelve a esperar.
        for tid in list(self._dentro):
            if tid not in vistos:
                del self._dentro[tid]

        self.max_en_zona = max(self.max_en_zona, en_zona)
        self.max_en_fila = max(self.max_en_fila, en_fila)
        hay = en_fila >= self.cfg.min_personas
        if hay:
            self.frames_con_fila += 1
        return EstadoFila(en_zona=en_zona, en_fila=en_fila, hay_fila=hay)

    @property
    def hubo_fila(self) -> bool:
        return self.frames_con_fila > 0

    @property
    def pct_frames_con_fila(self) -> float:
        if not self.frames_evaluados:
            return 0.0
        return round(100.0 * self.frames_con_fila / self.frames_evaluados, 2)
