"""Precision, recall y F1 de deteccion, contra ground truth etiquetado a mano.

El emparejamiento es el estandar de deteccion de objetos: una prediccion es un
acierto si su IoU con una caja real supera un umbral y esa caja real no fue ya
tomada por otra prediccion mejor. Las predicciones se ordenan por confianza
descendente, asi que la mejor se queda con la caja.

Sobre las cajas marcadas `dificil`: son personas sentadas y cortadas por el
borde, donde el propio etiquetado es dudoso. Se pueden excluir del calculo
(`ignorar_dificiles=True`) y ahi una prediccion que caiga sobre una de ellas no
cuenta ni como acierto ni como falso positivo, que es como las trata COCO. Sin
eso, el detector queda castigado por no encontrar lo que el anotador tampoco vio
bien.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Caja = tuple[float, float, float, float]


def iou(a: Caja, b: Caja) -> float:
    """Interseccion sobre union de dos cajas (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Conteos:
    """TP/FP/FN acumulados y las metricas que salen de ellos."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    ignoradas: int = 0  # predicciones que cayeron sobre una caja dificil
    meta: dict = field(default_factory=dict)

    def __add__(self, otro: "Conteos") -> "Conteos":
        return Conteos(self.tp + otro.tp, self.fp + otro.fp, self.fn + otro.fn,
                       self.ignoradas + otro.ignoradas)

    @property
    def precision(self) -> float:
        """De lo que el sistema dijo que era una persona, cuanto lo era."""
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        """De las personas que habia, cuantas encontro."""
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluar_frame(predicciones: list[tuple[Caja, float]],
                  reales: list[tuple[Caja, bool]],
                  iou_min: float = 0.5,
                  conf_min: float = 0.0,
                  ignorar_dificiles: bool = True) -> Conteos:
    """Empareja las predicciones de UN frame contra sus cajas reales.

    `predicciones` son (caja, score) y `reales` son (caja, es_dificil).
    """
    preds = sorted([p for p in predicciones if p[1] >= conf_min],
                   key=lambda p: -p[1])
    faciles = [c for c, dif in reales if not dif]
    dificiles = [c for c, dif in reales if dif]
    objetivo = faciles if ignorar_dificiles else faciles + dificiles

    tomadas: set[int] = set()
    c = Conteos()
    for caja, _ in preds:
        mejor, mejor_i = 0.0, -1
        for i, real in enumerate(objetivo):
            if i in tomadas:
                continue
            s = iou(caja, real)
            if s > mejor:
                mejor, mejor_i = s, i
        if mejor >= iou_min:
            tomadas.add(mejor_i)
            c.tp += 1
            continue
        # No matcheo ninguna caja contada. Si cae sobre una dificil que se decidio
        # ignorar, no se la castiga: el anotador tampoco estaba seguro.
        if ignorar_dificiles and any(iou(caja, d) >= iou_min for d in dificiles):
            c.ignoradas += 1
            continue
        c.fp += 1
    c.fn = len(objetivo) - len(tomadas)
    return c


def barrer_umbrales(por_frame: list[tuple[list, list]],
                    umbrales: list[float],
                    iou_min: float = 0.5,
                    ignorar_dificiles: bool = True) -> dict[float, Conteos]:
    """Corre evaluar_frame sobre todo el set, para cada umbral de confianza."""
    salida = {}
    for u in umbrales:
        total = Conteos()
        for preds, reales in por_frame:
            total = total + evaluar_frame(preds, reales, iou_min, u, ignorar_dificiles)
        salida[u] = total
    return salida
