"""Evalua el sistema contra el ground truth y barre umbrales de confianza.

Mide DOS cosas distintas, y la diferencia es el punto del reporte:

1. **El detector solo**, sobre los frames que el Nivel 1 dejo pasar. Es lo que
   suele reportarse como "precision del modelo".
2. **La cascada entera**, sobre TODOS los frames etiquetados. Un frame con una
   persona que MOG2 descarto es una persona perdida aunque el detector fuera
   perfecto: el gate no llega a ejecutarlo. Esa recall es la que le importa a
   quien despues va a confiar en el sistema.

El detector se corre con conf muy baja y el umbral se aplica DESPUES, sobre los
scores: asi el barrido es una sola pasada de GPU en vez de una por umbral.

Uso:
  python -m core.cascade.eval.run_eval
"""

from __future__ import annotations

import argparse
import json

import cv2

from core.cascade.config import load_cascade_config
from core.cascade.eval.exportar import escribir_metricas
from core.cascade.eval.metricas import barrer_umbrales
from core.cascade.stage2_person.runner import construir_detector

UMBRALES = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
IOU_MIN = 0.5
CONF_PISO = 0.05  # con que confianza se corre el detector antes de filtrar
# Tasa de paso del Nivel 1 medida sobre los 182 clips (stage2_person_report.md).
# Hace falta para corregir el sesgo del muestreo: la muestra etiquetada tiene a
# proposito mas frames descartados de los que hay naturalmente.
TASA_PASO_CORPUS = 0.6331


def _cargar_gt(cfg):
    p = cfg.corpus_dir / "groundtruth.json"
    if not p.exists():
        raise SystemExit(f"ABORTA: falta {p}. Etiquetar con core.cascade.eval.extraer")
    return json.loads(p.read_text(encoding="utf-8"))


def _leer_frame(cfg, gt_frame):
    ruta = next(iter(cfg.raw_dir.glob(f"{gt_frame['clip_id']}.*")), None)
    if ruta is None:
        return None
    cap = cv2.VideoCapture(str(ruta))
    cap.set(cv2.CAP_PROP_POS_FRAMES, gt_frame["frame_index"])
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _tabla(nombre: str, res: dict, L: list) -> None:
    L.append(f"### {nombre}")
    L.append("")
    L.append("| conf | TP | FP | FN | precision | recall | F1 |")
    L.append("|---|---|---|---|---|---|---|")
    mejor = max(res.items(), key=lambda kv: kv[1].f1)[0]
    for u, c in sorted(res.items()):
        marca = " **<-- mejor F1**" if u == mejor else ""
        L.append(f"| {u:.2f} | {c.tp} | {c.fp} | {c.fn} | {c.precision:.3f} | "
                 f"{c.recall:.3f} | {c.f1:.3f}{marca} |")
    L.append("")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evalua el cascade contra el ground truth")
    ap.add_argument("--iou", type=float, default=IOU_MIN)
    args = ap.parse_args()

    cfg = load_cascade_config()
    cv2.setNumThreads(cfg.cv_num_threads)
    gt = _cargar_gt(cfg)
    det = construir_detector(cfg)
    det.conf = CONF_PISO  # el umbral se aplica despues, sobre los scores
    if hasattr(det, "calentar"):
        det.calentar()
    prompts = list(cfg.person.prompts)

    # Una sola pasada del detector sobre cada frame etiquetado.
    solo_gate, todo = [], []
    n_leidos = 0
    for f in gt["frames"]:
        reales = [(tuple(p["bbox"]), p["dificil"]) for p in f["personas"]]
        img = _leer_frame(cfg, f)
        if img is None:
            print(f"AVISO: no se pudo leer {f['frame_id']}")
            continue
        n_leidos += 1
        alto, ancho = img.shape[:2]
        preds = [((d.bbox[0] / ancho, d.bbox[1] / alto,
                   d.bbox[2] / ancho, d.bbox[3] / alto), d.score)
                 for d in det.detect(img, prompts)]
        # La cascada solo ve los frames que paso el Nivel 1: en los demas, el
        # sistema no predice nada aunque el detector hubiera acertado.
        todo.append((preds if f["paso_nivel1"] else [], reales))
        if f["paso_nivel1"]:
            solo_gate.append((preds, reales))

    res_det = barrer_umbrales(solo_gate, UMBRALES, args.iou, True)
    res_sis = barrer_umbrales(todo, UMBRALES, args.iou, True)
    res_dif = barrer_umbrales(solo_gate, UMBRALES, args.iou, False)
    res_iou3 = barrer_umbrales(solo_gate, UMBRALES, 0.3, True)

    L: list[str] = []
    L.append("# Evaluacion contra ground truth — reporte")
    L.append("")
    L.append(f"protocolo: {cfg.person.backend} imgsz={cfg.person.imgsz} "
             f"device={cfg.person.device} | emparejamiento IoU>={args.iou} | "
             f"{n_leidos} frames, {gt['n_personas']} personas etiquetadas "
             f"({gt['n_dificiles']} dificiles)")
    L.append("")
    L.append("> El ground truth se etiqueto **a mano mirando los frames**, no con el "
             "modelo: si saliera del detector la evaluacion seria circular. Un solo "
             "anotador y sin segunda pasada, asi que los valores absolutos tienen el "
             "error del anotador adentro; lo que si es solido es la COMPARACION entre "
             "umbrales, que usa las mismas etiquetas para todos.")
    L.append("")

    L.append("## El detector solo")
    L.append("")
    L.append(f"Sobre los {len(solo_gate)} frames que el Nivel 1 dejo pasar. Es lo que "
             "normalmente se reporta como precision del modelo.")
    L.append("")
    _tabla(f"IoU >= {args.iou}, ignorando las cajas dificiles", res_det, L)

    L.append("## La cascada entera")
    L.append("")
    L.append(f"Sobre los {len(todo)} frames etiquetados, incluyendo los "
             f"{len(todo) - len(solo_gate)} que **el Nivel 1 descarto**. Ahi el sistema "
             "no predice nada, por definicion: el detector nunca llega a correr.")
    L.append("")
    _tabla(f"IoU >= {args.iou}, ignorando las cajas dificiles", res_sis, L)

    mejor_d = max(res_det.items(), key=lambda kv: kv[1].f1)
    mejor_s = max(res_sis.items(), key=lambda kv: kv[1].f1)
    L.append(f"El detector llega a F1 **{mejor_d[1].f1:.3f}** (recall "
             f"{mejor_d[1].recall:.3f}) y el sistema completo a F1 "
             f"**{mejor_s[1].f1:.3f}** (recall {mejor_s[1].recall:.3f}). "
             "**Toda esa diferencia es el gate**, no el modelo.")
    L.append("")

    # --- correccion del sesgo de muestreo -----------------------------------
    frac_paso_muestra = len(solo_gate) / len(todo) if todo else 0.0
    r_det = mejor_d[1].recall
    recall_corregida = TASA_PASO_CORPUS * r_det
    L.append("### Ojo con esta recall: la muestra esta sesgada a proposito")
    L.append("")
    L.append(f"El muestreo pidio frames descartados por el gate para poder medirlo, "
             f"asi que en la muestra pasa el **{100*frac_paso_muestra:.1f} %** de los "
             f"frames, contra el **{100*TASA_PASO_CORPUS:.1f} %** que pasa en los 182 "
             f"clips. La recall de {mejor_s[1].recall:.3f} describe la muestra, no el "
             f"corpus.")
    L.append("")
    L.append(f"Reponderando con la tasa real —y asumiendo que la gente se reparte igual "
             f"entre frames que pasan y frames que no— la recall del sistema sobre el "
             f"corpus seria del orden de **{recall_corregida:.3f}** "
             f"({TASA_PASO_CORPUS:.4f} x {r_det:.3f}).")
    L.append("")
    L.append("En los dos casos la lectura es la misma y es la que importa: **el techo "
             "de recall del sistema es la tasa de paso del Nivel 1**. Por bueno que sea "
             "el detector, no puede encontrar a nadie en un frame que nunca vio. El "
             "ahorro de GPU del Nivel 1 se paga en recall, y hasta ahora ese precio no "
             "estaba medido.")
    L.append("")

    L.append("## Sensibilidad de la medicion")
    L.append("")
    _tabla("Contando tambien las cajas dificiles (detector solo)", res_dif, L)
    _tabla("Aflojando el emparejamiento a IoU >= 0.3 (detector solo)", res_iou3, L)
    L.append("La segunda tabla dice cuanto de lo que se ve depende de la precision con "
             "que estan dibujadas las cajas a mano. Si F1 sube mucho al aflojar el IoU, "
             "el detector encuentra a la gente pero el anotador la encuadro distinto.")
    L.append("")

    L.append("## Veredicto")
    L.append("")
    L.append(f"Con el umbral que usa el pipeline (`conf={cfg.person.conf}`) el detector "
             f"da precision **{res_det[cfg.person.conf].precision:.3f}**, recall "
             f"**{res_det[cfg.person.conf].recall:.3f}**, F1 "
             f"**{res_det[cfg.person.conf].f1:.3f}**; el sistema completo, F1 "
             f"**{res_sis[cfg.person.conf].f1:.3f}**."
             if cfg.person.conf in res_det else
             f"El mejor F1 del detector es {mejor_d[1].f1:.3f} en conf={mejor_d[0]:.2f}.")
    L.append("")
    L.append(f"> **31 frames no alcanzan para fijar un umbral de produccion.** Son de "
             f"{len({f['clip_id'] for f in gt['frames']})} clips de la misma oficina, "
             "con las mismas personas y la misma camara. Sirven para ver la forma de la "
             "curva y para detectar un error grueso, no para elegir un valor definitivo.")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "eval_report.md").write_text("\n".join(L), encoding="utf-8")
    payload = {
        "protocolo": {"backend": cfg.person.backend, "imgsz": cfg.person.imgsz,
                      "iou_min": args.iou, "conf_piso": CONF_PISO,
                      "n_frames": n_leidos, "n_personas": gt["n_personas"]},
        "detector_solo": {str(u): c.__dict__ for u, c in res_det.items()},
        "sistema_completo": {str(u): c.__dict__ for u, c in res_sis.items()},
    }
    (cfg.results_dir / "eval_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    n = escribir_metricas(cfg.results_dir / "metricas_umbral.csv", payload)
    print(f"CSV: metricas_umbral.csv ({n} filas)")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
