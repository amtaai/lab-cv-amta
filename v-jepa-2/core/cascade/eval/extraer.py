"""Extrae una muestra de frames del corpus para etiquetar a mano.

El ground truth NO puede salir del modelo: seria circular. Este script solo
elige los frames y los escribe con una grilla de coordenadas encima, para que
quien etiquete pueda leer las cajas con alguna precision. Las etiquetas se
cargan despues, a mano, en corpus/groundtruth.json.

La muestra es ESTRATIFICADA por el veredicto del Nivel 1, y eso importa: si solo
se etiquetaran frames que el gate dejo pasar, la evaluacion mediria al detector
pero no al sistema. La recall de la cascada esta acotada por el Nivel 1 — un
frame con una persona que MOG2 descarto es una persona perdida, la vea o no el
detector despues.

Uso:
  python -m core.cascade.eval.extraer --n 40 --salida /tmp/gt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

from core.cascade.catalog.index import cargar_indice
from core.cascade.config import load_cascade_config
from core.cascade.stage1_motion.detector import MotionDetector

COLOR_GRILLA = (70, 70, 70)
COLOR_TEXTO = (0, 230, 255)


def dibujar_grilla(img, paso: float = 0.1):
    """Grilla cada `paso` del ancho/alto, rotulada en coordenadas normalizadas."""
    v = img.copy()
    alto, ancho = v.shape[:2]
    i = paso
    while i < 1.0:
        x, y = int(i * ancho), int(i * alto)
        cv2.line(v, (x, 0), (x, alto), COLOR_GRILLA, 1)
        cv2.line(v, (0, y), (ancho, y), COLOR_GRILLA, 1)
        cv2.putText(v, f"{i:.1f}", (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    COLOR_TEXTO, 1, cv2.LINE_AA)
        cv2.putText(v, f"{i:.1f}", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    COLOR_TEXTO, 1, cv2.LINE_AA)
        i += paso
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrae frames para etiquetar a mano")
    ap.add_argument("--n", type=int, default=40, help="cuantos frames extraer")
    ap.add_argument("--clips", type=int, default=16, help="de cuantos clips distintos")
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--semilla", type=int, default=20260818)
    args = ap.parse_args()

    cfg = load_cascade_config()
    cv2.setNumThreads(cfg.cv_num_threads)
    args.salida.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.semilla)

    clips = cargar_indice(cfg.corpus_dir)
    elegidos = rng.sample(clips, min(args.clips, len(clips)))
    por_clip = max(1, args.n // len(elegidos))

    muestras = []
    for c in elegidos:
        ruta = cfg.raw_dir / c.filename
        det = MotionDetector(cfg.motion)
        cap = cv2.VideoCapture(str(ruta))
        if not cap.isOpened():
            continue
        # Se recorre el clip entero clasificando cada frame por el veredicto del
        # Nivel 1, y recien despues se elige: asi la muestra puede incluir a
        # proposito frames que el gate descarto.
        con_mov, sin_mov, frames = [], [], {}
        idx = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            r = det.process(f)
            i, idx = idx, idx + 1
            if r.is_warmup:
                continue
            (con_mov if r.has_motion else sin_mov).append(i)
            frames[i] = f
        cap.release()

        # 2 de 3 con movimiento (los que el detector realmente ve) y 1 de 3 sin.
        n_con = max(1, (por_clip * 2) // 3)
        pick = rng.sample(con_mov, min(n_con, len(con_mov))) if con_mov else []
        pick += rng.sample(sin_mov, min(por_clip - len(pick), len(sin_mov))) if sin_mov else []
        for i in pick:
            nombre = f"{c.clip_id}__f{i:04d}"
            cv2.imwrite(str(args.salida / f"{nombre}.jpg"),
                        dibujar_grilla(frames[i]), [cv2.IMWRITE_JPEG_QUALITY, 92])
            muestras.append({
                "frame_id": nombre,
                "clip_id": c.clip_id,
                "frame_index": i,
                "width": frames[i].shape[1],
                "height": frames[i].shape[0],
                "paso_nivel1": i in con_mov,
            })

    (args.salida / "muestras.json").write_text(
        json.dumps({"semilla": args.semilla, "muestras": sorted(
            muestras, key=lambda m: m["frame_id"])}, indent=2), encoding="utf-8")
    n_con = sum(1 for m in muestras if m["paso_nivel1"])
    print(f"{len(muestras)} frames en {args.salida}")
    print(f"  {n_con} pasaron el Nivel 1 | {len(muestras) - n_con} los descarto el gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
