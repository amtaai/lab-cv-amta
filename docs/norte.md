# Norte del proyecto — de detecciones a datos en el tiempo

> Documento rector de `lab-cv-amta` (desde 2026-09-28). Todo lo que se construya en el lab
> se evalúa contra este norte. La línea froth queda **en pausa** (ver §6).

## 1. Qué se quiere

Un sistema al que se le da un **prompt en lenguaje natural** sobre una cámara
("personas que entran a la tienda A") y que, de forma continua:

1. **detecta / segmenta** los objetos del prompt (open-vocabulary: no hay clases fijas),
2. los **sigue** en el tiempo (identidad estable por objeto),
3. convierte ese seguimiento en **eventos** con significado (entró, salió, permaneció X s en
   la zona, hubo N simultáneos),
4. y **acumula esos eventos como datos tabulares en el tiempo**, sobre los que se puede
   hacer EDA y entrenar modelos de ML (conteo promedio por hora, picos, estacionalidad,
   anomalías, pronóstico de afluencia).

El producto **no son las máscaras ni las cajas**: son las **series temporales** que salen de
ellas. La segmentación es el medio; el dato es el fin. Es la misma lectura que ya se había
hecho en la línea froth ("el producto no son las máscaras: son series temporales en
ventanas móviles") — ahora es explícita y transversal al lab.

## 2. Pipeline de referencia

```
prompt ─┐
cámara ─┴─► [N1 movimiento] ─► [N2 detección open-vocab] ─► [tracking] ─► [N3 ReID]
                                   (YOLO-World / Grounding DINO /           │
                                    SAM 2/3 para máscaras)                   ▼
                                                                  [reglas de zona/línea]
                                                                             │
                     L1 detecciones ◄────────────────────────────────────────┤
                     L2 eventos     ◄────────────────────────────────────────┘
                     L3 agregados   ◄── ventanas temporales sobre L2
                                             │
                                             ▼
                                  EDA / ML (notebooks, modelos)
```

Principio heredado del lab: **no reescribir componentes que ya son SOTA**. El aporte es el
cableado prompt → eventos → datos, su medición (costo y precisión) y sus guardrails.

## 3. Contrato de datos (las tres capas)

Toda salida del sistema cae en una de estas capas. Es el contrato que consumen EDA y ML;
cambiarlo es una decisión explícita, no un efecto colateral.

### L1 — detecciones (una fila por objeto por frame analizado)

| campo | tipo | nota |
|---|---|---|
| `camara_id` | str | identidad estable de la fuente |
| `ts` | timestamp UTC | **absoluto**, no índice de frame |
| `frame` | int | índice dentro del clip/stream |
| `prompt` | str | concepto que produjo la detección |
| `track_id` | int | tras ReID |
| `x1,y1,x2,y2` | float | normalizadas [0,1] |
| `confianza` | float | |
| `confirmado` | bool | superó `min_hits` |
| `modelo` | str | backend + versión (reproducibilidad) |

Hoy existe como `v-jepa-2/results/detecciones.csv`, pero **sin `camara_id`, `ts`, `prompt`
ni `modelo`**, y con coordenadas en píxeles.

### L2 — eventos (una fila por hecho con significado)

| campo | tipo | nota |
|---|---|---|
| `camara_id`, `zona_id` | str | la zona es un polígono/línea con nombre ("tienda_A_puerta") |
| `ts` | timestamp UTC | |
| `tipo` | enum | `entrada`, `salida`, `cruce`, `permanencia`, `ocupacion` |
| `track_id` | int | |
| `prompt` | str | |
| `valor` | float | p. ej. segundos de permanencia |

Hoy el conteo por cruce de línea (`core/perception/conteo.py`) calcula entradas/salidas,
pero solo las entrega **agregadas por clip** (`resumen_clips.csv`), sin timestamp por evento.

### L3 — agregados temporales (una fila por ventana)

`camara_id, zona_id, prompt, ventana_inicio, ventana_fin, entradas, salidas,
ocupacion_media, ocupacion_max, permanencia_media, n_tracks`. Ventanas configurables
(1 min / 15 min / 1 h / 1 día). Es lo que responde "promedio de personas que entran a la
tienda A por hora".

### Almacenamiento

- Formato de intercambio: **Parquet** particionado por `camara_id` y fecha (EDA directo con
  pandas/polars/DuckDB).
- Servicio: la API ya existente (`core/api/`) expone L1/L2/L3 por job; a mediano plazo,
  Postgres/TimescaleDB para streams continuos.
- Las mediciones de costo siguen en `results/costs.db` (SQLite), aparte del dato de negocio.

## 4. Estado actual contra el norte (2026-09-28)

| Pieza | Estado | Dónde |
|---|---|---|
| Fuente de video (clip, archivo, RTSP) | ✅ | `core/api/`, `docker/` (simulador mediamtx) |
| N1 filtro de movimiento | ✅ medido | `core/cascade/stage1_motion/` |
| N2 detección por prompt | 🟡 YOLO-World acepta `prompts`; default es YOLO11 (clases COCO fijas) | `core/perception/detector.py` |
| Máscaras (SAM) en la cascada | ❌ no cableado (solo en notebooks) | `sam2_ov_*/` |
| Tracking | ✅ ByteTrack + registro `min_hits` | `core/perception/tracker.py` |
| ReID | ✅ DINOv2 + coseno, calibrado | `core/perception/reid.py` |
| Conteo por línea | ✅ entradas/salidas por clip | `core/perception/conteo.py` |
| Evaluación P/R/F1 con GT | ✅ | `core/cascade/eval/`, `results/eval_report.md` |
| Costo por etapa | ✅ | `core/cascade/cost/` |
| API REST | ✅ jobs en memoria | `core/api/` |
| **L1 con `ts`/`camara_id`/`prompt`** | ❌ | — |
| **L2 eventos con timestamp** | ❌ | — |
| **Zonas con nombre** (polígonos, no solo una línea) | ❌ | — |
| **L3 agregados por ventana** | ❌ | — |
| **Persistencia Parquet / DB** | ❌ (CSV sueltos en `results/`) | — |
| **EDA / ML sobre las series** | ❌ | — |

Lectura: la **percepción está resuelta y medida** (trabajo de Alex, semanas 1–2). El hueco
está entero en la **mitad de datos**: L1 enriquecida → L2 → L3 → EDA/ML.

## 5. Rol de cada pieza del lab bajo este norte

- `yolo_seg/`, `yolo_world/`, `sam2_ov_grounded/`, `sam2_ov_florence_2/`, `dino_v3/`: bancos
  de prueba de los **modelos candidatos para N2** (detección/segmentación por prompt) y
  ReID. Su valor se mide por cuánto mejoran L2 (precisión del conteo), no por la calidad
  visual de la máscara.
- `v-jepa-2/`: hoy aloja la plataforma (cascada + API). El encoder V-JEPA 2.1 queda como
  componente candidato (features de movimiento / novedad para eventos más ricos), no como
  el centro. El lazo de auto-corrección (VLM verificador + pseudo-labels) sigue siendo la
  meta de largo plazo para que el prompt "se afine solo" por cámara.
- `froth_gate/`: en pausa (§6).

## 6. Línea froth — en pausa

El GATE froth (C1/C2/C3 sobre IEEE, `froth_gate/`) queda **pendiente, sin trabajo activo**.
Estado congelado: C1 medido (F1 macro 91,91), C2 bloqueado por acceso gated a DINOv3, C3
sin correr. Los notebooks de Dorian (pipelines A y B) están en `froth_gate/`. Detalle en
`froth_gate/README.md` y `local_docs/froth/froth.md`. Nótese que froth encaja en el mismo
norte (burbujas → eventos → series), así que la infraestructura de datos que se construya
aquí le servirá cuando se retome.
