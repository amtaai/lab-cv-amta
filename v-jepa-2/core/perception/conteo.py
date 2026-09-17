"""Conteo de personas por cruce de linea.

Se apoya en `ultralytics.solutions.ObjectCounter`, que ya trae resuelto lo
dificil: decide cuando un track cruza la linea, en que direccion, y lleva la
lista de los ya contados para no contar a nadie dos veces.

Lo unico que se cambia de esa clase es DE DONDE saca los tracks. Tal como viene,
`extract_tracks()` corre su propio `model.track()`, o sea que detectaria por
segunda vez, saltandose el filtro de movimiento del Nivel 1 y dejando al medidor
de costo sin ver nada. Aca se sobreescribe ese metodo para que consuma los tracks
que el cascade YA calculo. Todo lo demas —historia de trayectorias, reglas de
cruce, dibujo— es de ultralytics sin tocar.

La linea va en coordenadas normalizadas [0,1] y no en pixeles: el corpus tiene
resoluciones mezcladas (640x480 y 1920x1080) y una geometria en pixeles caeria en
lugares distintos en cada una — el mismo error que ya se habia corregido en el
umbral de area del Nivel 1.

NOTA: hubo tambien una deteccion de filas sobre una ROI (`QueueManager`). Se
quito porque presuponia una zona de espera —una caja de comercio— que los
datasets de este proyecto no tienen: la ROI terminaba siendo un rectangulo
arbitrario y su numero no describia nada. Si aparece material de un local con
area de caja, se reconstruye desde `ultralytics.solutions.QueueManager` con el
mismo mixin de tracks externos que usa el contador.
"""

from __future__ import annotations

import numpy as np

from core.perception.detector import YOLO11_SEG, resolver_pesos
from core.types import Detection

CLASE_PERSONA = 0


def a_pixeles(puntos, ancho: int, alto: int) -> list[tuple[int, int]]:
    """Pasa una geometria normalizada [0,1] a pixeles del frame."""
    return [(int(round(x * ancho)), int(round(y * alto))) for x, y in puntos]


def utiles(dets: list[Detection]) -> list[Detection]:
    """Las detecciones que se pueden usar para contar.

    Se piden las dos cosas. Sin `track_id` no hay a quien contar —y ByteTrack no
    siempre asigna uno, asi que la caja llega con track_id None—; sin confirmar,
    un parpadeo de un frame contaria como una persona que entro al local.
    """
    return [d for d in dets
            if d.track_id is not None and d.meta.get("confirmado", True)]


class _MixinTracksExternos:
    """Hace que una solucion de ultralytics consuma tracks ya calculados.

    `extract_tracks` es el unico punto donde las soluciones llaman al modelo. Al
    sobreescribirlo, el resto de la clase funciona igual pero sin inferir.
    """

    def cargar(self, dets: list[Detection]) -> None:
        """Guarda las detecciones que se van a usar en el proximo count/process."""
        self._dets = dets

    def extract_tracks(self, im0):  # noqa: D102 - reemplaza la de ultralytics
        dets = getattr(self, "_dets", [])
        self.boxes = [list(d.bbox) for d in dets]
        self.clss = [CLASE_PERSONA] * len(dets)
        self.track_ids = [int(d.track_id) for d in dets]
        self.track_data = None


def _kwargs_base(region_px, line_width: int) -> dict:
    """Argumentos comunes de las soluciones de ultralytics.

    `model` va explicito aunque nunca se use: `BaseSolution.__init__` instancia un
    YOLO si o si y, sin decirle cual, BAJA yolo11n.pt al directorio actual —o sea
    que ensucia el repo y depende de la red. Apuntandolo a los pesos que ya estan
    en la imagen, no descarga nada.
    """
    return dict(region=region_px, show=False, line_width=line_width,
                model=resolver_pesos(YOLO11_SEG[0]), verbose=False)


def _preparar(solucion, region_px) -> None:
    """Suelta el modelo que la solucion cargo y que no vamos a usar.

    `extract_tracks` esta sobreescrito, asi que ese YOLO nunca infiere: se lo
    libera para no dejar pesos ocupando memoria. `names` si hace falta, porque
    ultralytics lo usa para etiquetar las cajas que dibuja.
    """
    solucion.model = None
    solucion.names = {CLASE_PERSONA: "person"}
    solucion.region = region_px


class ContadorPersonas:
    """Punto 3: cuando una persona entra, sale y cuando se la contabiliza.

    Las tres reglas, explicitas:

    - **Entra**: el centroide del track cruza la linea de conteo y el movimiento
      va en el sentido positivo (a la derecha si la linea es vertical, hacia
      abajo si es horizontal). Lo decide `ObjectCounter.count_objects`.
    - **Sale**: mismo cruce, sentido contrario.
    - **Se contabiliza**: una sola vez por `track_id`. ultralytics lleva
      `counted_ids` y no vuelve a contar un track ya contado, asi que una persona
      que va y viene sobre la linea no infla el numero.

    Ademas hace falta que el track este confirmado por el registro (`min_hits`):
    un parpadeo de un frame no es una persona.
    """

    def __init__(self, linea_norm, ancho: int, alto: int, line_width: int = 2):
        from ultralytics.solutions import ObjectCounter

        class _Contador(_MixinTracksExternos, ObjectCounter):
            pass

        px = a_pixeles(linea_norm, ancho, alto)
        self._sol = _Contador(**_kwargs_base(px, line_width))
        _preparar(self._sol, px)
        self.linea_px = px

    @property
    def entradas(self) -> int:
        return int(self._sol.in_count)

    @property
    def salidas(self) -> int:
        return int(self._sol.out_count)

    @property
    def contados(self) -> int:
        """Tracks distintos que cruzaron la linea alguna vez."""
        return len(self._sol.counted_ids)

    def actualizar(self, frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
        """Procesa un frame y devuelve el frame anotado por ultralytics."""
        self._sol.cargar(utiles(dets))
        return self._sol.count(frame)
