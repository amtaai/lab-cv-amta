"""Dibuja el resultado del Nivel 2 sobre un clip, para poder mirarlo.

El reporte da numeros; esto da la imagen. Sirve sobre todo para auditar el
seguimiento: si una persona cambia de ID a mitad del clip se ve al toque, y ese
error no se nota en ningun agregado.

Uso:
  python -m core.cascade.stage2_person.visualizar --clip <clip_id>
  python -m core.cascade.stage2_person.visualizar --clip <clip_id> --video
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from core.cascade.config import load_cascade_config
from core.cascade.stage1_motion.detector import MotionDetector
from core.cascade.stage2_person.runner import construir_detector
from core.perception.tracker import Tracker
from core.perception.conteo import a_pixeles
from core.perception.filas import DetectorFila, FilaConfig, cargar_rois
from core.perception.reid import ReIdConfig, ReIdentificador

# Un color fijo por track_id: el mismo ID siempre sale del mismo color, asi el
# cambio de ID se ve como un cambio de color y no hay que leer el numero.
PALETA = [(66, 214, 66), (66, 135, 245), (245, 176, 66), (200, 66, 245),
          (66, 245, 227), (245, 66, 108), (150, 245, 66), (245, 245, 66)]


def color_de(track_id: int | None) -> tuple[int, int, int]:
    """Color estable por ID. Sin ID (deteccion sin seguir) va gris."""
    if track_id is None:
        return (160, 160, 160)
    return PALETA[track_id % len(PALETA)]


def caja_int(d) -> tuple[int, int, int, int]:
    """El bbox de Detection viene en float; para dibujar hace falta en px enteros."""
    x1, y1, x2, y2 = d.bbox
    return int(x1), int(y1), int(x2), int(y2)


COLOR_LINEA = (0, 215, 255)  # amarillo: la linea de conteo
COLOR_ROI = (255, 128, 0)  # azul: la zona de espera, si el clip tiene ROI


def dibujar_linea(v, cfg, roi=None) -> None:
    """Pinta la linea de conteo y, si la hay, la zona de espera. In place."""
    alto, ancho = v.shape[:2]
    linea = a_pixeles(cfg.conteo.linea, ancho, alto)
    cv2.line(v, linea[0], linea[1], COLOR_LINEA, 2)
    if roi:
        import numpy as np

        cv2.polylines(v, [np.array(a_pixeles(roi, ancho, alto), dtype=np.int32)],
                      True, COLOR_ROI, 2)


def anotar(frame, dets, idx: int, estado: str, cfg=None, roi=None):
    """Dibuja las cajas con su ID y confianza sobre una copia del frame."""
    v = frame.copy()
    if cfg is not None:
        dibujar_linea(v, cfg, roi)
    for d in dets:
        x1, y1, x2, y2 = caja_int(d)
        c = color_de(d.track_id)
        grosor = 2 if d.meta.get("confirmado") else 1  # sin confirmar = linea fina
        cv2.rectangle(v, (x1, y1), (x2, y2), c, grosor)
        etiqueta = (f"id{d.track_id} {d.score:.2f}" if d.track_id is not None
                    else f"{d.label} {d.score:.2f}")
        cv2.putText(v, etiqueta, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    cv2.putText(v, f"f{idx} {estado}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Dibuja detecciones y IDs de un clip")
    ap.add_argument("--clip", required=True, help="clip_id del corpus (sin extension)")
    ap.add_argument("--video", action="store_true", help="escribe un mp4 en vez de un jpg")
    ap.add_argument("--muestras", type=int, default=6, help="frames del contacto (modo jpg)")
    args = ap.parse_args()

    cfg = load_cascade_config()
    ruta = next(iter(cfg.raw_dir.glob(f"{args.clip}.*")), None)
    if ruta is None:
        print(f"ABORTA: no existe el clip {args.clip} en {cfg.raw_dir}")
        return 1

    cv2.setNumThreads(cfg.cv_num_threads)
    det_mov = MotionDetector(cfg.motion)
    det_per = construir_detector(cfg)
    reg = Tracker(max_age=cfg.track.max_age, min_hits=cfg.track.min_hits)
    prompts = list(cfg.person.prompts)

    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # El Nivel 3 va aca tambien: si no, el visualizador muestra los IDs partidos
    # de ByteTrack y no lo que el pipeline realmente produce.
    reid = (ReIdentificador(ReIdConfig(), fps=fps, device=cfg.person.device)
            if cfg.reid.activo else None)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # La zona de espera es de la camara: sale de corpus/rois.json por clip_id.
    roi = cargar_rois(cfg.filas.rois_path).get(args.clip) if cfg.filas.activo else None
    fila = (DetectorFila(roi, ancho, alto, fps=fps, cfg=FilaConfig())
            if roi else None)

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    salida = cfg.results_dir / f"vis_{args.clip}.{'mp4' if args.video else 'jpg'}"
    vw = (cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
          if args.video else None)
    # En modo jpg se guardan unos pocos frames repartidos a lo largo del clip.
    quiero = {int(i * total / (args.muestras + 1)) for i in range(1, args.muestras + 1)}
    tiras, idx = [], 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i, idx = idx, idx + 1
        r = det_mov.process(frame)
        if r.is_warmup or not r.has_motion:
            estado = "warmup" if r.is_warmup else "descartado por Nivel 1"
            if vw is not None:
                vw.write(anotar(frame, [], i, estado, cfg, roi))
            elif i in quiero:
                tiras.append(anotar(frame, [], i, estado, cfg, roi))
            continue
        dets = det_per.track(frame, prompts)
        if reid is not None:
            dets = reid.procesar(frame, dets)
        dets = reg.update(dets)
        estado = f"Nivel 2: {len(dets)} cajas"
        if fila is not None:
            e = fila.actualizar(frame.copy(), dets)
            estado += f" | zona {e.en_zona} esperando {e.en_fila}"
            if e.hay_fila:
                estado += " FILA"
        if vw is not None:
            vw.write(anotar(frame, dets, i, estado, cfg, roi))
        elif i in quiero:
            tiras.append(anotar(frame, dets, i, estado, cfg, roi))
    cap.release()

    if vw is not None:
        vw.release()
    else:
        if not tiras:
            print("ABORTA: no se pudo leer ningun frame")
            return 1
        cv2.imwrite(str(salida), np.hstack(tiras), [cv2.IMWRITE_JPEG_QUALITY, 85])

    extra = f" | {reid.n_reenganches} reenganches del Nivel 3" if reid else ""
    if fila is not None:
        extra += (f" | fila max {fila.max_en_fila}, "
                  f"{fila.pct_frames_con_fila:.1f}% del tiempo")
    print(f"OK {salida} | {idx} frames | {reg.unique_count} personas unicas{extra}")
    print("Linea gruesa = track confirmado. Un cambio de color es un cambio de ID.")
    print("Amarillo = linea de conteo. Azul = zona de espera (si el clip tiene ROI).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
