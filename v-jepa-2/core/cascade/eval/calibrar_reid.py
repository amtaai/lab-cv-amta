"""Calibra el umbral de similitud del Nivel 3, sin etiquetar nada a mano.

EL PROBLEMA. El umbral decide cuando dos recortes son "la misma persona". Estaba
en 0.65 porque lo elegi a ojo, y de ese numero depende todo el reenganche: muy
alto no reengancha nada, muy bajo fusiona personas distintas.

DE DONDE SALEN LOS PARES. Del propio seguimiento, sin anotar nada:

- POSITIVOS (misma persona): dos recortes del MISMO track_id separados por varios
  frames. Dentro de un tramo en que ByteTrack no perdio el track, es la misma
  persona con alta confianza.
- NEGATIVOS (personas distintas): dos recortes de track_id distintos presentes en
  el MISMO frame. Nadie esta en dos lugares a la vez, asi que son distintas con
  certeza.

Los negativos son los solidos; los positivos pueden traer algun cambio de ID de
ByteTrack colado, lo que solo puede EMPEORAR la separacion medida. O sea que el
numero que sale es conservador.

Ademas compara recortar el rectangulo crudo contra borrar el fondo con la mascara
del detector, que ya viene calculada y hasta hace poco se tiraba.

Uso:
  python -m core.cascade.eval.calibrar_reid --clips retail_
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random

import cv2
import numpy as np

from core.cascade.catalog.index import cargar_indice
from core.cascade.config import load_cascade_config
from core.cascade.stage1_motion.detector import MotionDetector
from core.cascade.stage2_person.runner import construir_detector
from core.perception.reid import ReIdConfig, ReIdentificador

SEP_MIN, SEP_MAX = 8, 60  # separacion en frames entre los dos recortes de un positivo
MAX_PARES = 600
MAX_POR_TRACK = 24  # cuantas apariciones se guardan de cada track
LOTE = 32
UMBRALES = np.arange(0.30, 0.96, 0.01)


def _recolectar(cfg, clips, det, reid):
    """Recorre los clips y guarda, por track, los VECTORES de cada aparicion.

    Se embebe sobre la marcha y se tiran los recortes. Guardarlos para embeberlos
    despues parece mas simple y es lo que hice primero: son ~25.000 detecciones por
    dos variantes a 240 KB cada una, mas de 10 GB, y lo mato el OOM killer. Un
    vector son 1,5 KB.
    """
    crudo = ReIdentificador(ReIdConfig(usar_mascara=False), device="cpu")
    con_m = ReIdentificador(ReIdConfig(usar_mascara=True), device="cpu")
    muestras = collections.defaultdict(list)   # (clip, tid) -> [(frame_idx, v_crudo, v_mask)]
    por_frame = collections.defaultdict(list)  # (clip, frame_idx) -> [tid]
    buffer = []  # (clave, frame_idx, recorte_crudo, recorte_mask) pendientes de embeber

    def volcar():
        if not buffer:
            return
        va = reid._embeber([b[2] for b in buffer])
        vb = reid._embeber([b[3] for b in buffer])
        for (clave, idx, _, _), x, y in zip(buffer, va, vb):
            muestras[clave].append((idx, x, y))
        buffer.clear()

    for c in clips:
        dm = MotionDetector(cfg.motion)
        det.reset()
        cap = cv2.VideoCapture(str(cfg.raw_dir / c.filename))
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            r = dm.process(f)
            idx, i = i, i + 1
            if r.is_warmup or not r.has_motion:
                continue
            for d in det.track(f, list(cfg.person.prompts)):
                if d.track_id is None:
                    continue
                clave = (c.clip_id, d.track_id)
                por_frame[(c.clip_id, idx)].append(d.track_id)
                # Se guardan pocas apariciones por track: alcanzan para armar pares
                # y evita que un track largo domine la muestra.
                if len(muestras[clave]) + sum(1 for b in buffer if b[0] == clave) \
                        >= MAX_POR_TRACK:
                    continue
                a, b = crudo._recorte(f, d), con_m._recorte(f, d)
                if a is None or b is None:
                    continue
                buffer.append((clave, idx, a, b))
                if len(buffer) >= LOTE:
                    volcar()
        cap.release()
        volcar()
        print(f"  {c.clip_id}: {sum(1 for k in muestras if k[0] == c.clip_id)} tracks",
              flush=True)
    return muestras, por_frame


def _armar_pares(muestras, por_frame, rng):
    pos, neg = [], []
    for lista in muestras.values():
        for _ in range(min(12, len(lista))):
            if len(lista) < 2:
                break
            a, b = rng.sample(range(len(lista)), 2)
            if SEP_MIN <= abs(lista[a][0] - lista[b][0]) <= SEP_MAX:
                pos.append((lista[a], lista[b]))
    for (clip, idx), tids in por_frame.items():
        for t1, t2 in itertools.combinations(set(tids), 2):
            l1 = [x for x in muestras[(clip, t1)] if x[0] == idx]
            l2 = [x for x in muestras[(clip, t2)] if x[0] == idx]
            if l1 and l2:
                neg.append((l1[0], l2[0]))
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:MAX_PARES], neg[:MAX_PARES]


def _similitudes(pares, col):
    """Coseno entre los dos vectores de cada par. Ya vienen L2-normalizados."""
    return np.array([float(np.dot(a[col], b[col])) for a, b in pares])


def _mejor_umbral(sp, sn):
    """Umbral que maximiza aciertos, y cuanto acierta de cada lado."""
    mejor = (0.0, 0.0)
    for u in UMBRALES:
        acc = ((sp >= u).sum() + (sn < u).sum()) / (len(sp) + len(sn))
        if acc > mejor[1]:
            mejor = (float(u), float(acc))
    u = mejor[0]
    return u, mejor[1], float((sp >= u).mean()), float((sn < u).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibra el umbral de ReID")
    ap.add_argument("--clips", default="retail_", help="filtro por clip_id")
    ap.add_argument("--semilla", type=int, default=20260821)
    args = ap.parse_args()

    cfg = load_cascade_config()
    cv2.setNumThreads(cfg.cv_num_threads)
    rng = random.Random(args.semilla)
    clips = [c for c in cargar_indice(cfg.corpus_dir) if args.clips in c.clip_id]
    if not clips:
        print(f"ABORTA: ningun clip contiene '{args.clips}'")
        return 1

    det = construir_detector(cfg)
    det.calentar()
    reid = ReIdentificador(ReIdConfig(), device=cfg.person.device)
    muestras, por_frame = _recolectar(cfg, clips, det, reid)
    pos, neg = _armar_pares(muestras, por_frame, rng)
    print(f"\npares: {len(pos)} positivos (misma persona), "
          f"{len(neg)} negativos (personas distintas)")
    if not pos or not neg:
        print("ABORTA: no se pudieron armar pares")
        return 1

    filas = []
    print(f"\n{'variante':16s}{'pos media':>11s}{'neg media':>11s}{'separa':>9s}"
          f"{'umbral':>9s}{'acierto':>9s}{'reeng.ok':>10s}{'no fusiona':>12s}")
    for nombre, col in (("recorte crudo", 1), ("fondo borrado", 2)):
        sp, sn = _similitudes(pos, col), _similitudes(neg, col)
        u, acc, tpr, tnr = _mejor_umbral(sp, sn)
        print(f"{nombre:16s}{sp.mean():11.3f}{sn.mean():11.3f}"
              f"{sp.mean() - sn.mean():9.3f}{u:9.2f}{acc * 100:8.1f}%"
              f"{tpr * 100:9.1f}%{tnr * 100:11.1f}%", flush=True)
        filas.append({"variante": nombre, "pos_media": float(sp.mean()),
                      "neg_media": float(sn.mean()), "umbral": u, "acierto": acc,
                      "recupera_mismas": tpr, "rechaza_distintas": tnr,
                      "pos_p10": float(np.percentile(sp, 10)),
                      "neg_p90": float(np.percentile(sn, 90))})

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "reid_calibracion.json").write_text(json.dumps({
        "modelo": reid.cfg.modelo, "n_pos": len(pos), "n_neg": len(neg),
        "clips": [c.clip_id for c in clips], "semilla": args.semilla,
        "variantes": filas,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\npos_p10 / neg_p90 (donde se pisan las colas):")
    for f in filas:
        print(f"  {f['variante']:16s} pos_p10={f['pos_p10']:.3f}  neg_p90={f['neg_p90']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
