"""Medicion objetiva de la iluminacion de un clip.

Etiquetar la iluminacion a ojo sobre cientos de clips no escala y no es
auditable. Aca se mide sobre los pixeles y se clasifica con una regla explicita.

Las metricas crudas quedan en ClipMeta.meta, asi que la regla se puede cambiar
despues y reclasificar el corpus entero con `ingest --remedir` sin volver a
copiar nada.

NOTA sobre la metrica: la primera version usaba p95-p5 como "contraste". Medido
sobre el corpus real, ese numero da 165-246 para TODOS los clips, porque el rango
dinamico es alto en cualquier escena real — clasificaba los 182 clips como MIXED
y el eje no informaba nada. Lo que de verdad distingue una escena de iluminacion
mixta es tener zonas quemadas Y zonas aplastadas EN EL MISMO frame, que es lo que
se mide ahora.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.cascade.catalog.schema import Lighting

# Umbrales absolutos, fijados mirando la distribucion medida sobre el corpus.
LUMA_DIM_MAX = 110.0  # luma media 0-255 por debajo de esto: poca luz
FRAC_EXTREMO_MIN = 0.5  # % de pixeles quemados y aplastados para llamarlo mixta
UMBRAL_QUEMADO = 245  # pixel >= esto: highlight sin informacion
UMBRAL_OSCURO = 15  # pixel <= esto: sombra sin informacion
N_MUESTRAS = 12  # frames muestreados uniformemente a lo largo del clip


def clasificar_iluminacion(luma_media: float, frac_quemado_pct: float,
                           frac_oscuro_pct: float) -> Lighting:
    """Regla de clasificacion. Separada de la medicion para testearla sola.

    DIM gana sobre MIXED: si la escena es oscura, el problema dominante para el
    detector es la falta de luz, no el rango dinamico.
    """
    if luma_media < LUMA_DIM_MAX:
        return Lighting.DIM
    if frac_quemado_pct >= FRAC_EXTREMO_MIN and frac_oscuro_pct >= FRAC_EXTREMO_MIN:
        return Lighting.MIXED
    return Lighting.BRIGHT


def metricas_de_frames(frames: list) -> dict:
    """luma media y % de pixeles quemados / aplastados, promediados sobre frames."""
    if not frames:
        return {"luma_media": 0.0, "frac_quemado_pct": 0.0, "frac_oscuro_pct": 0.0,
                "n_frames_muestreados": 0}
    lumas, quemados, oscuros = [], [], []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        lumas.append(float(g.mean()))
        quemados.append(float((g >= UMBRAL_QUEMADO).mean()) * 100.0)
        oscuros.append(float((g <= UMBRAL_OSCURO).mean()) * 100.0)
    return {
        "luma_media": round(float(np.mean(lumas)), 2),
        "frac_quemado_pct": round(float(np.mean(quemados)), 3),
        "frac_oscuro_pct": round(float(np.mean(oscuros)), 3),
        "n_frames_muestreados": len(frames),
    }


def _muestrear(path, n_muestras: int) -> list:
    """Devuelve hasta n_muestras frames repartidos uniformemente por el clip."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        if total > 0:
            # Uniforme: el arranque y el final no representan todo el clip.
            indices = np.linspace(0, max(total - 1, 0), num=min(n_muestras, max(total, 1)))
            for i in np.unique(indices.astype(int)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        if not frames:
            # Fallback: algunos contenedores mienten en FRAME_COUNT o no dejan seek.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(n_muestras):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        return frames
    finally:
        cap.release()


def medir_iluminacion(path, n_muestras: int = N_MUESTRAS) -> dict:
    """Muestrea el clip y devuelve las metricas + la etiqueta derivada."""
    frames = _muestrear(path, n_muestras)
    if not frames:
        return {"luma_media": 0.0, "frac_quemado_pct": 0.0, "frac_oscuro_pct": 0.0,
                "n_frames_muestreados": 0, "lighting": Lighting.DIM.value,
                "error": f"no se pudo leer {path}"}
    m = metricas_de_frames(frames)
    m["lighting"] = clasificar_iluminacion(
        m["luma_media"], m["frac_quemado_pct"], m["frac_oscuro_pct"]
    ).value
    return m
