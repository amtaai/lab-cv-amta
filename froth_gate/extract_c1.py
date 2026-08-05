# GATE froth (FO1) — extraccion de features C1 (geometrico industrial) sobre IEEE.
# Corre LOCAL (CPU, multiprocessing). Lee los 4 zips on-the-fly (no extrae a disco).
# Salida: froth_gate/results/c1_features.parquet — una fila por secuencia (2386),
# con features por-frame agregadas (mean+std sobre los 12 frames) + deltas temporales.
#
# Pre-registro (froth.md §8.2/§8.4): parametros fijados a ojo sobre UNA secuencia de
# muestra ANTES de correr cualquier probe; no se tunean contra el test.
# C1-IEEE = subconjunto por-frame + deltas (sin tracking: a 0.4 s no es computable).
import io
import json
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as sstats
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from skimage.filters import gaussian, sobel
from skimage.registration import phase_cross_correlation
from skimage.segmentation import watershed

IEEE_DIR = Path(r"C:\amta-lab-cv\data_froth\IEEE\Dataset_ _Flotation Froth Sequence Images_")
OUT_DIR = Path(r"C:\amta-lab-cv\froth_gate\results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GLCM_DISTANCES = [1, 4]
GLCM_LEVELS = 32
GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation"]
WATERSHED_MIN_DIST = 8      # px entre picos (marcadores de burbuja)
WATERSHED_SIGMA = 2.0       # suavizado previo
FFT_BANDS = 4


def features_frame(g):
    """27 features geometrico/texturales de un frame grayscale float [0,1]."""
    f = {}
    # intensidad (7)
    f["int_mean"], f["int_std"] = float(g.mean()), float(g.std())
    f["int_skew"] = float(sstats.skew(g.ravel()))
    f["int_kurt"] = float(sstats.kurtosis(g.ravel()))
    p10, p50, p90 = np.percentile(g, [10, 50, 90])
    f["int_p10"], f["int_p50"], f["int_p90"] = float(p10), float(p50), float(p90)
    # GLCM (8): 4 props x 2 distancias, promediadas sobre 4 angulos
    q = np.clip((g * (GLCM_LEVELS - 1)).astype(np.uint8), 0, GLCM_LEVELS - 1)
    glcm = graycomatrix(q, distances=GLCM_DISTANCES,
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    for prop in GLCM_PROPS:
        v = graycoprops(glcm, prop)          # (n_dist, n_ang)
        for di, d in enumerate(GLCM_DISTANCES):
            f[f"glcm_{prop}_d{d}"] = float(v[di].mean())
    # gradiente (2)
    sob = sobel(g)
    f["sobel_mean"], f["sobel_std"] = float(sob.mean()), float(sob.std())
    # FFT radial (4): fraccion de energia por banda de frecuencia espacial
    F = np.abs(np.fft.rfft2(g - g.mean())) ** 2
    fy = np.fft.fftfreq(g.shape[0])[:, None]
    fx = np.fft.rfftfreq(g.shape[1])[None, :]
    r = np.sqrt(fy**2 + fx**2)
    total = F.sum() + 1e-12
    edges = np.linspace(0, 0.5, FFT_BANDS + 1)
    for b in range(FFT_BANDS):
        f[f"fft_band{b}"] = float(F[(r >= edges[b]) & (r < edges[b + 1])].sum() / total)
    # watershed (6): proxy clasico de BSD (pre-DL; burbujas = picos de brillo)
    gs = gaussian(g, sigma=WATERSHED_SIGMA)
    peaks = peak_local_max(gs, min_distance=WATERSHED_MIN_DIST, exclude_border=False)
    markers = np.zeros(g.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(sob, markers)
    f["ws_n_regions"] = float(labels.max())
    # areas excluyendo regiones que tocan el borde (regla DeepX: parciales sesgan BSD)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    areas = np.bincount(labels.ravel())[1:]          # por region (1..n)
    keep = np.ones(len(areas), dtype=bool)
    keep[border[border > 0] - 1] = False
    a = areas[keep].astype(np.float64)
    if len(a) >= 3:
        ap10, ap50, ap90 = np.percentile(a, [10, 50, 90])
        d = 2.0 * np.sqrt(a / np.pi)
        f["ws_area_p10"], f["ws_area_p50"], f["ws_area_p90"] = ap10, ap50, ap90
        f["ws_d32_px"] = float((d**3).sum() / ((d**2).sum() + 1e-12))
    else:
        f["ws_area_p10"] = f["ws_area_p50"] = f["ws_area_p90"] = f["ws_d32_px"] = 0.0
    return f


def deltas_secuencia(frames, order=None):
    """Features temporales sobre pares consecutivos (sensibles al orden)."""
    if order is not None:
        frames = [frames[i] for i in order]
    diffs, corrs, shifts = [], [], []
    for a, b in zip(frames[:-1], frames[1:]):
        diffs.append(np.abs(a - b).mean())
        corrs.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        shift, _, _ = phase_cross_correlation(a, b, upsample_factor=1, normalization="phase")
        shifts.append(np.hypot(*shift))
    return {"dt_absdiff_mean": float(np.mean(diffs)),
            "dt_corr_mean": float(np.mean(corrs)),
            "dt_corr_std": float(np.std(corrs)),
            "dt_shift_mean": float(np.mean(shifts)),
            "dt_shift_std": float(np.std(shifts))}


def procesar_secuencia(args):
    zip_path, seq_id, clase, rng_seed = args
    t0 = time.perf_counter()
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if re.search(rf"/{seq_id}/\d+\.jpg$", n))
        frames = [np.asarray(Image.open(io.BytesIO(z.read(n))).convert("L"), dtype=np.float32) / 255.0
                  for n in names]
    per_frame = [features_frame(g) for g in frames]
    row = {"clase": clase, "seq_id": seq_id, "n_frames": len(frames)}
    keys = per_frame[0].keys()
    for k in keys:
        vals = np.array([pf[k] for pf in per_frame])
        row[f"{k}_mean"], row[f"{k}_std"] = float(vals.mean()), float(vals.std())
    row.update(deltas_secuencia(frames))
    # control de permutacion temporal (froth.md §8.4): mismas deltas con orden permutado
    perm = np.random.RandomState(rng_seed).permutation(len(frames))
    row.update({f"PERM_{k}": v for k, v in deltas_secuencia(frames, order=perm).items()})
    row["t_extract_s"] = time.perf_counter() - t0
    return row


def main():
    tareas = []
    for k, zp in enumerate(sorted(IEEE_DIR.glob("*.zip")), 1):
        with zipfile.ZipFile(zp) as z:
            seqs = sorted({m.group(1) for n in z.namelist()
                           if (m := re.search(r"/(\d+)/\d+\.jpg$", n))})
        print(f"clase {k}: {zp.name} -> {len(seqs)} secuencias")
        tareas += [(str(zp), s, k, 42_000 + k * 10_000 + int(s)) for s in seqs]
    print(f"total: {len(tareas)} secuencias")

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for i, row in enumerate(ex.map(procesar_secuencia, tareas, chunksize=8)):
            rows.append(row)
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"{i+1}/{len(tareas)} | {el:.0f}s | ETA {el/(i+1)*(len(tareas)-i-1):.0f}s",
                      flush=True)
    df = pd.DataFrame(rows).sort_values(["clase", "seq_id"]).reset_index(drop=True)
    out = OUT_DIR / "c1_features.parquet"
    df.to_parquet(out, index=False)
    meta = {"n_seqs": len(df), "por_clase": df["clase"].value_counts().to_dict(),
            "n_features": len([c for c in df.columns if c not in
                               ("clase", "seq_id", "n_frames", "t_extract_s")]),
            "t_total_s": time.time() - t0,
            "t_por_seq_s": float(df["t_extract_s"].mean()),
            "params": {"glcm_distances": GLCM_DISTANCES, "glcm_levels": GLCM_LEVELS,
                       "ws_min_dist": WATERSHED_MIN_DIST, "ws_sigma": WATERSHED_SIGMA}}
    (OUT_DIR / "c1_extract_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
