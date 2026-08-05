# froth_gate/ — GATE froth (FO1) sobre el dataset IEEE

Experimento decisivo de `local_docs/froth/froth.md` §8: comparación C1/C2/C3 para
clasificación de condición operativa (4 clases, 2.386 secuencias de 12 frames @ 0.4 s).
Umbrales GO/NO-GO **congelados antes de mirar resultados** (froth.md §8.3).

## Piezas y dónde corre cada una (topología del repo: LOCAL sin GPU, Colab = heavy)

| Pieza | Corre en | Qué hace |
|---|---|---|
| `extract_c1.py` | **LOCAL** (CPU, ~50 min con 10 procesos) | C1 geométrico industrial: GLCM, intensidad, Sobel, FFT radial, watershed-BSD (D32 px, regla anti-parciales de DeepX), deltas temporales (abs-diff, corr, phase correlation) + variante PERM. → `results/c1_features.parquet` |
| `gate_froth_ieee.ipynb` | **Colab T4** (~35–50 min) | C2 (DINOv3 ViT-S y ViT-B, HF gated) y C3 (V-JEPA 2.1 ViT-B 384, patrón smoke test: checkpoint manual + parche RoPE + fp16) + **pasada C3 con orden de frames permutado** (control §8.4). → npz en `Amta_lab/outputs/gate_froth/` |
| `gate_analysis.py` | **LOCAL** | junta todo → k-fold estratificado por secuencia (k=5, seed 42), probe LogReg idéntico, F1 macro, correlación de scores, fusiones, control de permutación, diagnóstico de leakage por bloques → `results/gate_report.md` con veredicto |

## Flujo

1. `python extract_c1.py` (una vez; local).
2. Subir los 4 zips IEEE a Drive `Amta_lab/data/ieee_froth/` (una vez, manual).
3. Ejecutar `gate_froth_ieee.ipynb` en Colab T4 (VS Code → New Colab Server). Al final
   persiste los npz a Drive. **Remove Server al terminar.**
4. Bajar `Amta_lab/outputs/gate_froth/*` a `froth_gate/results/`.
5. `python gate_analysis.py` → veredicto.

## Decisiones de protocolo (pre-registradas)

- **Entrada**: cada condición usa su preprocessing nativo sobre los frames originales
  692×518 (C1: resolución original; C2: 224²; C3: 384² center-crop). La resolución
  efectiva es parte del punto de operación y se reporta con los costos. Nadie ve frames
  que otro no vea.
- **Split**: k-fold estratificado POR SECUENCIA (nunca frames de una secuencia en train
  y test). Diagnóstico adicional de leakage: split por bloques contiguos de `seq_id`
  (las secuencias son cronológicas; vecinas pueden compartir condiciones de captura).
- **Probe**: StandardScaler + LogisticRegression (C=1, seed 42) idéntico para las tres.
- **Permutación**: semilla determinista por secuencia (42000 + clase·10000 + seq_id),
  idéntica en `extract_c1.py` (deltas C1) y en el notebook (clips C3).
- **Salvedad congelada** (froth.md §6/§8.1): a Δt=0.4 s los frames de las clases Ⅰ–Ⅲ
  están decorrelacionados (corr≈0, medido) — el GATE mide "textura temporal a 0.4 s",
  no movimiento continuo. El control de permutación decide si una eventual ventaja de
  C3 es genuinamente temporal.

## Limitaciones conocidas

- C1 no incluye métricas de tracking (imposibles a 0.4 s): es el subconjunto por-frame
  + deltas. C1-completo (burst/renewal/coalescencia) solo existe en video continuo (T6).
- Confound clase↔tiempo: las 4 clases vienen de períodos de captura distintos (4 zips);
  no controlable con este dataset.
- Paridad de costo asimétrica: C1 es CPU, C2/C3 GPU — se reporta latencia por hardware
  y FLOPs teóricos de los ViT; la curva de Pareto completa (§8.2) queda para la
  iteración 2 si el resultado lo amerita.
