# GATE froth (FO1) — análisis y veredicto GO/NO-GO.
# Corre LOCAL (CPU). Junta C1 (c1_features.parquet, de extract_c1.py) con C2/C3
# (npz de gate_froth_ieee.ipynb, bajados de Drive Amta_lab/outputs/gate_froth/).
# Protocolo y umbrales CONGELADOS en local_docs/froth/froth.md §8.3 ANTES de mirar
# resultados: X=5 pts F1 macro, delta=2, corr<0.5, fusión >=+2, k-fold por secuencia k=5.
# Funciona parcial: reporta lo que haya y lista lo que falta.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RES = Path(__file__).parent / "results"
SEED = 42
K = 5
UMBRAL_X = 5.0      # pts F1 macro: margen GO
UMBRAL_DELTA = 2.0  # pts: empate = NO-GO
UMBRAL_CORR = 0.5   # complementariedad (GO exige <0.5; >0.7 = redundante NO-GO)
UMBRAL_FUSION = 2.0 # pts: ganancia aditiva mínima de la fusión


def probe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=1.0, random_state=SEED))


def evaluar(X, y, splits):
    """F1 macro por fold + scores out-of-fold (log-prob de la clase verdadera) + aciertos."""
    f1s, baccs = [], []
    oof_score = np.zeros(len(y))
    oof_ok = np.zeros(len(y), dtype=bool)
    cm = np.zeros((4, 4), dtype=int)
    for tr, te in splits:
        clf = probe().fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        logp = clf.predict_log_proba(X[te])
        f1s.append(f1_score(y[te], pred, average="macro") * 100)
        baccs.append(balanced_accuracy_score(y[te], pred) * 100)
        cls_index = {c: i for i, c in enumerate(clf.classes_)}
        oof_score[te] = logp[np.arange(len(te)), [cls_index[c] for c in y[te]]]
        oof_ok[te] = pred == y[te]
        cm += confusion_matrix(y[te], pred, labels=[1, 2, 3, 4])
    return {"f1_folds": np.round(f1s, 2), "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "bacc": float(np.mean(baccs)), "cm": cm, "oof_score": oof_score, "oof_ok": oof_ok}


def cargar_condiciones():
    conds, notas = {}, []
    df = pd.read_parquet(RES / "c1_features.parquet")
    df["key"] = df["clase"].astype(str) + "_" + df["seq_id"].astype(str)
    df = df.sort_values("key").reset_index(drop=True)
    y = df["clase"].to_numpy()
    keys = df["key"].to_numpy()
    fcols = [c for c in df.columns if c not in ("clase", "seq_id", "n_frames", "t_extract_s", "key")
             and not c.startswith("PERM_")]
    pcols = [c.replace("dt_", "PERM_dt_") if c.startswith("dt_") else c for c in fcols]
    conds["C1_geom"] = df[fcols].to_numpy()
    conds["C1_geom_PERM"] = df[pcols].to_numpy()
    notas.append(f"C1: {len(fcols)} features ({RES/'c1_features.parquet'})")

    for tag, fn in [("C2_dinov3_vits", "c2_dinov3_vits.npz"), ("C2_dinov3_vitb", "c2_dinov3_vitb.npz"),
                    ("C3_vjepa21", "c3_vjepa21_vitb.npz"), ("C3_vjepa21_PERM", "c3_vjepa21_vitb_PERM.npz")]:
        p = RES / fn
        if not p.exists():
            notas.append(f"{tag}: FALTA {fn} (correr gate_froth_ieee.ipynb en Colab y bajar de Drive)")
            continue
        z = np.load(p, allow_pickle=True)
        zkey = np.char.add(np.char.add(z["clase"].astype(str), "_"), z["seq_id"].astype(str))
        orden = np.argsort(zkey)
        zkey = zkey[orden]
        assert np.array_equal(zkey, keys), f"{tag}: keys no coinciden con C1 — revisar índice"
        conds[tag] = z["emb"][orden]
        notas.append(f"{tag}: dim {conds[tag].shape[1]} ({fn})")

    # ---- exclusion de secuencias con frames corruptos (declarada, no silenciosa)
    # 13 secuencias del dataset IEEE contienen al menos un frame COMPLETAMENTE NEGRO
    # (std = 0, un unico valor de pixel). En C1 eso deja NaN: skew/kurtosis y la
    # correlacion entre frames no estan definidas cuando la varianza es cero.
    # C2/C3 no producen NaN, pero SI producen un embedding muy distintivo para una
    # imagen negra — y las 13 son de la clase 4, asi que el frame corrupto es un
    # predictor casi perfecto de esa clase. Dejarlas dentro inflaria el F1 de las tres
    # condiciones con un artefacto del dataset en vez de semantica de froth.
    # Se excluyen de TODAS las condiciones (nadie evalua sobre secuencias que otro no ve).
    mask = ~df[[c for c in df.columns if c != "key"]].isna().any(axis=1).to_numpy()
    n_out = int((~mask).sum())
    if n_out:
        excluidas = keys[~mask]
        notas.append(f"EXCLUIDAS {n_out} secuencias con frames negros (std=0): "
                     f"{', '.join(excluidas)}")
        conds = {k: v[mask] for k, v in conds.items()}
        y, keys = y[mask], keys[mask]
        notas.append(f"n efectivo: {len(y)} secuencias "
                     f"({dict(zip(*np.unique(y, return_counts=True)))})")
    return conds, y, keys, notas


def main():
    conds, y, keys, notas = cargar_condiciones()
    print("\n".join(notas))
    rng = np.random.RandomState(SEED)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # ---- evaluación por condición
    res = {}
    for nombre, X in conds.items():
        res[nombre] = evaluar(X, y, splits)
        print(f"{nombre:22s} F1 macro = {res[nombre]['f1']:.2f} ± {res[nombre]['f1_std']:.2f} "
              f"| bacc = {res[nombre]['bacc']:.2f} | folds: {res[nombre]['f1_folds']}")

    # ---- diagnóstico de leakage temporal: split por bloques contiguos de seq_id
    fold_bloque = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]        # keys ordenadas -> seq_id ascendente dentro de clase
        fold_bloque[idx] = np.floor(np.linspace(0, K - 1e-9, len(idx))).astype(int)
    splits_bloque = [(np.where(fold_bloque != f)[0], np.where(fold_bloque == f)[0]) for f in range(K)]
    print("\n[diagnóstico leakage] split por bloques contiguos de seq_id (no decisorio):")
    diag_bloque = {}
    for nombre in [n for n in ("C1_geom", "C2_dinov3_vitb", "C3_vjepa21") if n in conds]:
        r = evaluar(conds[nombre], y, splits_bloque)
        diag_bloque[nombre] = r["f1"]
        print(f"  {nombre:20s} F1(bloques) = {r['f1']:.2f}  (vs {res[nombre]['f1']:.2f} aleatorio; "
              f"caída grande => leakage temporal suave en el k-fold)")

    lineas = ["# GATE froth — reporte", "",
              f"protocolo: k-fold estratificado por secuencia k={K}, seed={SEED}, probe=LogReg+scaler",
              ""]
    lineas += ["## F1 macro por condición", ""]
    for nombre, r in res.items():
        lineas.append(f"- **{nombre}**: {r['f1']:.2f} ± {r['f1_std']:.2f} (folds {list(r['f1_folds'])})")

    # ---- veredicto (solo si están las tres condiciones)
    tiene_todo = all(k in res for k in ("C1_geom", "C3_vjepa21", "C3_vjepa21_PERM")) and \
        any(k in res for k in ("C2_dinov3_vits", "C2_dinov3_vitb"))
    if not tiene_todo:
        lineas += ["", "## Veredicto: PENDIENTE — faltan condiciones (ver arriba)"]
        print("\nVeredicto: PENDIENTE — faltan C2/C3 (correr el notebook Colab).")
    else:
        c2best = max((k for k in ("C2_dinov3_vits", "C2_dinov3_vitb") if k in res),
                     key=lambda k: res[k]["f1"])
        f1_c1, f1_c2, f1_c3 = res["C1_geom"]["f1"], res[c2best]["f1"], res["C3_vjepa21"]["f1"]
        rival = max(f1_c1, f1_c2)
        margen = f1_c3 - rival
        folds_gana = int((res["C3_vjepa21"]["f1_folds"] >
                          np.maximum(res["C1_geom"]["f1_folds"], res[c2best]["f1_folds"])).sum())
        # complementariedad: corr de scores oof + fusión temprana (concat estandarizado)
        corr_c1 = float(np.corrcoef(res["C3_vjepa21"]["oof_score"], res["C1_geom"]["oof_score"])[0, 1])
        corr_c2 = float(np.corrcoef(res["C3_vjepa21"]["oof_score"], res[c2best]["oof_score"])[0, 1])
        fus = {}
        for otro in ("C1_geom", c2best):
            # concat crudo: el StandardScaler del probe ya escala DENTRO de cada fold.
            # Pre-escalar acá con fit sobre el dataset completo era redundante y metía
            # leakage del test en la normalizacion.
            Xf = np.concatenate([conds[otro], conds["C3_vjepa21"]], axis=1)
            rf = evaluar(Xf, y, splits)
            fus[otro] = rf["f1"] - max(res[otro]["f1"], f1_c3)
            lineas.append(f"- fusión C3⊕{otro}: {rf['f1']:.2f} (Δ vs mejor solo: {fus[otro]:+.2f})")
        # control de permutación
        caida_perm = f1_c3 - res["C3_vjepa21_PERM"]["f1"]

        go_a = margen >= UMBRAL_X and folds_gana >= 4
        go_b = (max(corr_c1, corr_c2) < UMBRAL_CORR) and (max(fus.values()) >= UMBRAL_FUSION)
        nogo_empate = margen <= UMBRAL_DELTA
        nogo_redund = max(corr_c1, corr_c2) > 0.7
        nogo_perm = caida_perm <= 0.5  # la ventaja no se degrada al romper el orden temporal

        lineas += ["", "## Criterios (congelados en froth.md §8.3)", "",
                   f"- margen C3 − max(C1,C2) = {margen:+.2f} pts (X≥{UMBRAL_X}; empate ±{UMBRAL_DELTA}) — C3 gana en {folds_gana}/5 folds",
                   f"- corr(score C3, C1) = {corr_c1:.3f} · corr(score C3, {c2best}) = {corr_c2:.3f} (GO exige <{UMBRAL_CORR}; >0.7 redundante)",
                   f"- control permutación: F1 C3 {f1_c3:.2f} → PERM {res['C3_vjepa21_PERM']['f1']:.2f} (caída {caida_perm:+.2f}; sin caída ⇒ señal NO temporal)",
                   ""]
        if go_a and go_b and not nogo_redund and not nogo_perm:
            ver = "GO"
        elif nogo_empate or nogo_redund or nogo_perm or not go_a:
            motivos = []
            if nogo_empate: motivos.append(f"empate/derrota (margen {margen:+.2f} ≤ {UMBRAL_DELTA})")
            elif not go_a: motivos.append(f"margen insuficiente ({margen:+.2f} < {UMBRAL_X} o folds {folds_gana}/5 < 4)")
            if nogo_redund: motivos.append("redundancia (corr > 0.7)")
            if nogo_perm: motivos.append("la ventaja NO se degrada bajo permutación (señal no temporal)")
            if not go_b and not nogo_empate: motivos.append("sin complementariedad suficiente")
            ver = "NO-GO — " + "; ".join(motivos)
        else:
            # zona gris de §8.3: gana con margen pero la complementariedad no alcanza
            # el criterio GO (0.5 ≤ corr ≤ 0.7, o fusión < +2) — no se firma sin revisar
            ver = ("INDETERMINADO — C3 supera el margen pero sin complementariedad "
                   f"suficiente (corr max {max(corr_c1, corr_c2):.3f}, fusión max "
                   f"{max(fus.values()):+.2f}); revisar froth.md §8.3 antes de firmar")
        lineas += [f"## Veredicto: {ver}", "",
                   "(pendiente de la curva de Pareto de costo: ver costs.json y froth.md §8.2 antes de firmar)"]
        print(f"\n=== VEREDICTO: {ver} ===")

    costs = RES / "costs.json"
    if costs.exists():
        lineas += ["", "## Costos (Colab T4)", "```json", costs.read_text(), "```"]
    (RES / "gate_report.md").write_text("\n".join(lineas), encoding="utf-8")
    print(f"\nreporte: {RES/'gate_report.md'}")


if __name__ == "__main__":
    main()
