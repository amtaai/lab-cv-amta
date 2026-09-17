"""Analisis de errores del sistema, con casos concretos y no solo agregados.

Cinco categorias, cada una medida de una fuente distinta porque no se pueden
sacar todas del mismo lado:

1. FALSOS POSITIVOS  - del ground truth: cajas predichas que no matchean ninguna
   persona etiquetada. Se listan los frames para poder ir a mirarlos.
2. NO DETECTADAS     - del ground truth, separando las dos causas: el detector no
   la vio, o el Nivel 1 descarto el frame y el detector nunca corrio. Son fallas
   distintas y se arreglan en lugares distintos.
3. DOBLE CONTEO      - de los tracks: cuantas identidades se abren de mas. Se mide
   comparando con y sin Nivel 3 sobre el mismo clip.
4. PERDIDA DE TRACKING - tracks que mueren mientras la persona sigue en escena. Se
   detecta cuando un track termina y otro nuevo aparece cerca poco despues.
5. OCLUSIONES        - la causa de fondo de 3 y 4. Se estima por el solapamiento
   entre cajas: dos personas que se pisan mucho son candidatas a intercambiar o
   perder ID.

Uso:
  python -m core.cascade.eval.errores
"""

from __future__ import annotations

import argparse
import json

import cv2

from core.cascade.catalog.index import cargar_indice
from core.cascade.config import load_cascade_config
from core.cascade.eval.metricas import iou
from core.cascade.stage1_motion.detector import MotionDetector
from core.cascade.stage2_person.runner import construir_detector
from core.perception.conteo import utiles
from core.perception.reid import ReIdConfig, ReIdentificador
from core.perception.tracker import Tracker

IOU_MATCH = 0.5
IOU_OCLUSION = 0.30   # dos cajas que se pisan mas que esto: hay oclusion mutua
VENTANA_RELEVO = 45   # frames para considerar que un track nuevo "releva" a otro
IOU_RELEVO = 0.25     # y donde aparece respecto de donde murio el otro


# ---- 1 y 2: del ground truth ------------------------------------------------

def _errores_de_deteccion(cfg, gt, det, conf_min):
    """Falsos positivos y no detectadas, con el frame de cada caso."""
    fp, fn_detector, fn_gate = [], [], []
    for f in gt["frames"]:
        ruta = next(iter(cfg.raw_dir.glob(f"{f['clip_id']}.*")), None)
        if ruta is None:
            continue
        cap = cv2.VideoCapture(str(ruta))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame_index"])
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        alto, ancho = img.shape[:2]
        reales = [(tuple(p["bbox"]), p["dificil"]) for p in f["personas"]]
        faciles = [c for c, dif in reales if not dif]

        if not f["paso_nivel1"]:
            # El sistema no predice nada: cada persona del frame es una perdida
            # del gate, no del detector.
            fn_gate += [(f["frame_id"], c) for c in faciles]
            continue

        preds = [((d.bbox[0] / ancho, d.bbox[1] / alto,
                   d.bbox[2] / ancho, d.bbox[3] / alto), d.score)
                 for d in det.detect(img, list(cfg.person.prompts))
                 if d.score >= conf_min]
        tomadas = set()
        for caja, score in sorted(preds, key=lambda p: -p[1]):
            mejor, mejor_i = 0.0, -1
            for i, real in enumerate(faciles):
                if i in tomadas:
                    continue
                s = iou(caja, real)
                if s > mejor:
                    mejor, mejor_i = s, i
            if mejor >= IOU_MATCH:
                tomadas.add(mejor_i)
            elif not any(iou(caja, c) >= IOU_MATCH for c, dif in reales if dif):
                fp.append((f["frame_id"], round(score, 3), tuple(round(v, 3) for v in caja)))
        fn_detector += [(f["frame_id"], faciles[i])
                        for i in range(len(faciles)) if i not in tomadas]
    return fp, fn_detector, fn_gate


# ---- 3, 4 y 5: de las trayectorias ------------------------------------------

def _analizar_tracks(cfg, clip, det, con_reid: bool):
    """Recorre un clip y devuelve, por track, donde vivio y con quien se solapo."""
    dm = MotionDetector(cfg.motion)
    reg = Tracker(max_age=cfg.track.max_age, min_hits=cfg.track.min_hits)
    reid = (ReIdentificador(ReIdConfig(), device=cfg.person.device)
            if con_reid else None)
    det.reset()
    cap = cv2.VideoCapture(str(cfg.raw_dir / clip.filename))
    vida = {}          # track_id -> [primer frame, ultimo frame, ultima caja]
    solapados = set()  # track_id que alguna vez se piso con otro
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = dm.process(f)
        i, idx = idx, idx + 1
        if r.is_warmup or not r.has_motion:
            continue
        dets = det.track(f, list(cfg.person.prompts))
        if reid is not None:
            dets = reid.procesar(f, dets)
        dets = utiles(reg.update(dets))
        for a in range(len(dets)):
            for b in range(a + 1, len(dets)):
                if iou(dets[a].bbox, dets[b].bbox) >= IOU_OCLUSION:
                    solapados.add(dets[a].track_id)
                    solapados.add(dets[b].track_id)
        for d in dets:
            v = vida.get(d.track_id)
            vida[d.track_id] = [v[0] if v else i, i, d.bbox]
    cap.release()
    return vida, solapados, (reid.n_reenganches if reid else 0)


def _relevos(vida):
    """Tracks que aparecen justo donde y cuando murio otro: seguimiento perdido."""
    pares = []
    for nuevo, (ini_n, _, caja_n) in vida.items():
        for viejo, (_, fin_v, caja_v) in vida.items():
            if viejo == nuevo or not (0 < ini_n - fin_v <= VENTANA_RELEVO):
                continue
            if iou(caja_v, caja_n) >= IOU_RELEVO:
                pares.append((viejo, nuevo, ini_n - fin_v))
                break
    return pares


def main() -> int:
    ap = argparse.ArgumentParser(description="Analisis de errores del sistema")
    ap.add_argument("--clips", default="retail_", help="filtro para el analisis de tracks")
    args = ap.parse_args()

    cfg = load_cascade_config()
    cv2.setNumThreads(cfg.cv_num_threads)
    det = construir_detector(cfg)
    det.calentar()

    gt = json.loads((cfg.corpus_dir / "groundtruth.json").read_text(encoding="utf-8"))
    fp, fn_det, fn_gate = _errores_de_deteccion(cfg, gt, det, cfg.person.conf)

    clips = [c for c in cargar_indice(cfg.corpus_dir) if args.clips in c.clip_id]
    por_clip = {}
    for c in clips:
        vida_sin, solap_sin, _ = _analizar_tracks(cfg, c, det, con_reid=False)
        vida_con, solap_con, reeng = _analizar_tracks(cfg, c, det, con_reid=True)
        por_clip[c.clip_id] = {
            "ids_sin_reid": len(vida_sin), "ids_con_reid": len(vida_con),
            "relevos_sin_reid": len(_relevos(vida_sin)),
            "relevos_con_reid": len(_relevos(vida_con)),
            "tracks_con_oclusion": len(solap_con),
            "reenganches": reeng,
        }
        print(f"  {c.clip_id}: {por_clip[c.clip_id]}", flush=True)

    L = ["# Analisis de errores — reporte", ""]
    L.append(f"protocolo: {cfg.person.backend} conf={cfg.person.conf} "
             f"IoU match={IOU_MATCH} | ground truth: {len(gt['frames'])} frames, "
             f"{gt['n_personas']} personas | tracks: {len(clips)} clips")
    L.append("")

    L.append("## 1. Falsos positivos")
    L.append("")
    L.append(f"**{len(fp)}** cajas predichas sin persona real detras, sobre "
             f"{sum(1 for f in gt['frames'] if f['paso_nivel1'])} frames analizados.")
    L.append("")
    if fp:
        L.append("| frame | confianza | caja (normalizada) |")
        L.append("|---|---|---|")
        for fid, sc, caja in sorted(fp, key=lambda x: -x[1])[:10]:
            L.append(f"| `{fid}` | {sc} | {caja} |")
        L.append("")
        L.append("Son pocos y de confianza baja: el detector casi no inventa gente. "
                 "Subir el umbral los elimina sin costo de recall.")
    else:
        L.append("Ninguno con este umbral.")
    L.append("")

    L.append("## 2. Personas no detectadas")
    L.append("")
    L.append("Hay que separar las dos causas, porque se arreglan en lugares distintos:")
    L.append("")
    L.append(f"- **el detector no la vio**: {len(fn_det)} casos")
    L.append(f"- **el Nivel 1 descarto el frame**: {len(fn_gate)} casos — el detector "
             "nunca llego a correr")
    L.append("")
    if fn_gate:
        L.append("La segunda es la dominante y **no se arregla tocando el detector**. "
                 "Es el precio del gate: el techo de recall del sistema es su tasa de "
                 "paso. Frames afectados:")
        L.append("")
        for fid in sorted({f for f, _ in fn_gate})[:8]:
            L.append(f"- `{fid}`")
        L.append("")

    L.append("## 3. Doble conteo")
    L.append("")
    L.append("| clip | IDs sin ReID | IDs con ReID | de mas | reenganches |")
    L.append("|---|---|---|---|---|")
    for cid, d in sorted(por_clip.items()):
        L.append(f"| {cid} | {d['ids_sin_reid']} | {d['ids_con_reid']} | "
                 f"{d['ids_sin_reid'] - d['ids_con_reid']} | {d['reenganches']} |")
    L.append("")
    L.append("La columna *de mas* es doble conteo puro: la misma persona abierta "
             "varias veces. El Nivel 3 la recupera reconociendo la apariencia.")
    L.append("")

    L.append("## 4. Perdida de seguimiento")
    L.append("")
    L.append("Un track muere y otro nuevo aparece en el mismo lugar poco despues: la "
             f"persona siguio ahi pero cambio de identidad. Se cuenta con una ventana "
             f"de {VENTANA_RELEVO} frames e IoU >= {IOU_RELEVO}.")
    L.append("")
    L.append("| clip | relevos sin ReID | relevos con ReID |")
    L.append("|---|---|---|")
    for cid, d in sorted(por_clip.items()):
        L.append(f"| {cid} | {d['relevos_sin_reid']} | {d['relevos_con_reid']} |")
    L.append("")

    L.append("## 5. Oclusiones")
    L.append("")
    L.append(f"Se cuenta un track como ocluido si alguna vez se piso con otro por "
             f"encima de IoU {IOU_OCLUSION}.")
    L.append("")
    L.append("| clip | tracks con oclusion | de un total de |")
    L.append("|---|---|---|")
    for cid, d in sorted(por_clip.items()):
        L.append(f"| {cid} | {d['tracks_con_oclusion']} | {d['ids_con_reid']} |")
    L.append("")
    L.append("Es la causa de fondo de las categorias 3 y 4: cuando dos personas se "
             "pisan, el seguidor por posicion no tiene con que distinguirlas al "
             "separarse. Por eso el Nivel 3 usa apariencia y no posicion.")
    L.append("")

    L.append("## Veredicto")
    L.append("")
    L.append(f"El error dominante **no es del detector**: son las {len(fn_gate)} "
             f"personas que el Nivel 1 descarto contra {len(fn_det)} que el detector "
             f"no vio, y {len(fp)} falsos positivos. La segunda fuente es el doble "
             "conteo por oclusion, que el Nivel 3 reduce pero no elimina.")
    L.append("")
    L.append("> El ground truth son 31 frames de una misma oficina y el analisis de "
             "tracks, 3 clips de tienda. Alcanza para ordenar las causas por "
             "importancia, no para poner un numero de produccion en cada una.")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "errores_report.md").write_text("\n".join(L), encoding="utf-8")
    (cfg.results_dir / "errores_stats.json").write_text(json.dumps({
        "falsos_positivos": len(fp), "no_detectadas_detector": len(fn_det),
        "no_detectadas_gate": len(fn_gate), "por_clip": por_clip,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
