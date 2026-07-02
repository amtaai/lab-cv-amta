# 03 - Estado del arte y chequeo de novedad

> Sintesis de la investigacion de prior-art para el proyecto: ¿que ya existe, que
> tan cerca esta de lo nuestro, y que queda como hueco defendible? Este documento es
> el insumo factual del veredicto ([`04_veredicto_novedad.md`](04_veredicto_novedad.md)):
> aca estan las fuentes; alla esta la decision.
>
> **Validacion de fuentes (2026-06-17).** Todos los IDs de arxiv de este documento
> fueron verificados contra arxiv.org el 2026-06-17. **Resultado: los 18 IDs son
> reales y su titulo/tema coincide con lo afirmado.** Los que originalmente venian
> de la busqueda sin confirmar quedan marcados **(verificado 2026-06-17)** como rastro
> de auditoria. Se corrigio **un error**: YOLO-World figuraba como `2405.14874` (que en
> realidad es "Open-Vocabulary Object Detectors: Robustness Challenges under
> Distribution Shifts", Chhipa et al.); el ID correcto es **`2401.17270`**. Aun asi:
> antes de citar en un paper, abrir el PDF y confirmar el claim especifico, no solo la
> existencia del ID.

---

## 1. Como leer este documento

Mapeamos el espacio en cinco bloques: (A) los **componentes** que usamos como ladrillos,
(B) el **patron** "VLM/foundation-model verifica pseudo-labels para auto-entrenar un
detector" — que es lo mas parecido a nuestro lazo y por eso el riesgo de novedad #1,
(C) la **adaptacion en cámara / scene-specific**, (D) los **backbones** que compiten por
ser la fuente de la senal (V-JEPA 2.1 vs DINOv3), y (E) la **conclusion de huecos**.

Cada entrada trae: que hace, que tan cerca esta de nosotros, y la diferencia que nos
deja (o nos quita) espacio.

---

## 2. Bloque A — Componentes (ladrillos, no contribucion)

Ninguno de estos es aporte; son infraestructura del estado del arte que ensamblamos.
Se listan para que un colaborador sepa exactamente que esta resuelto aguas arriba.

| Componente | arxiv | Que aporta | Rol en el proyecto |
|---|---|---|---|
| **V-JEPA 2** | `2506.09985` | World-model de video auto-supervisado (JEPA); embeddings espaciotemporales; NO da cajas ni mascaras | `VJepa2Encoder` (fuente candidata de la senal de novedad) |
| **V-JEPA 2.1** | `2603.14482` **(verificado 2026-06-17)** | Mejora dense features via Dense Predictive Loss + self-supervision jerarquica | el encoder que de hecho usamos (entrypoints `vjepa2_1_*`) |
| **SAM 2** | `2408.00714` | Segmentacion promptable en video con memoria temporal y propagacion de mascaras | `Sam2Segmenter` |
| **Grounding DINO** | `2303.05499` | Deteccion open-vocabulary desde texto | `OpenVocabDetector` (arranque por lenguaje natural) |
| **YOLO-World** | `2405.14874` | OVD en tiempo real | `OpenVocabDetector` (tier rapido) |
| **Mask2Former** | `2112.01527` | Segmentacion unificada (semantic/instance/panoptic) | referencia para mascaras densas |
| **ByteTrack / FairMOT** | `2110.06864` / `2004.01888` | Asociacion temporal e IDs (MOT) | `Tracker` |
| **Florence-2 / InternVL** | `2311.06242` / `2312.14238` | VLMs con grounding; verificadores VQA | `VlmVerifier` |

**Lectura clave:** que estos ocho existan y funcionen es exactamente por que el sistema
integrado, por si solo, **no es publicable como novedad**. Ensamblar APIs maduras es
ingenieria. La novedad tiene que venir de algo que ninguno de estos provee.

---

## 3. Bloque B — El patron "verificar pseudo-labels para auto-entrenar" (riesgo #1)

Este es el bloque que mas se parece a nuestro lazo. Si no nos diferenciamos de aca,
no hay paper. **Esta publicado 5+ veces.**

| Trabajo | arxiv | Que hace | Distancia a nosotros |
|---|---|---|---|
| **Autodistill** | (Roboflow, sin arxiv canonico) | base-model → pseudo-labels → target-model; literalmente "NL → destila un detector" | **Es nuestra Fase 3-4 ya empaquetada.** One-shot, sin loop de feedback ni anti-drift de largo horizonte. |
| **VLM-PL** | `2403.05346` | VLM como pseudo-labeler para deteccion incremental | Mismo patron VLM-verifica; foco en incremental, no en drift temporal en stream continuo. |
| **SAS-Det** | `2308.06412` | Self-training con auto-etiquetado para OVD; calibra confianza | Baseline directo de self-training OVD. No usa senal temporal de world-model. |
| **DST-Det** | `2310.01393` | Self-training dinamico para deteccion open-vocab | Idem: pseudo-labels + filtrado; sin eje temporal/predictivo. |
| **"The Detector Teaches Itself"** | `2605.03642` **(verificado 2026-06-17)** | Self-teaching detector+VLM (DAT) sobre COCO/LVIS, afina el backbone del VLM, <0.8M params | Cubre el **patron** self-teaching, pero **estatico** (no temporal, no stream, no drift de largo horizonte): no ocupa nuestro hueco. Riesgo controlado. |
| **Co-teaching VLM** | `2511.09955` **(verificado 2026-06-17)** | Per-object co-teaching entre 2 YOLO para filtrar pseudo-labels ruidosos de VLM (KITTI/ACDC/BDD) | Anti-drift por co-teaching; alternativa/competencia a nuestro VLM-circuit-breaker, pero estatica. |
| **SAM2Auto** | `2506.07850` **(verificado 2026-06-17)** | Auto-anotacion de video (SMART-OD + FLASH), sin intervencion humana | **El vecino mas cercano en arquitectura.** Pero es integracion para anotar, no una tesis sobre una senal. Nos sirve para decir "eso es integracion, lo nuestro es la senal". |

**Implicacion para el relato (ver §2 de [`01_vision_y_plan.md`](01_vision_y_plan.md)):**
el lazo, el VLM-verificador y "NL → detector" **NO se afirman como aporte**. Se citan
como prior-art que ya resolvio esa parte.

---

## 4. Bloque C — Adaptacion en cámara / scene-specific

El caso "cámara fija que se especializa a su escena" tampoco es nuevo.

| Trabajo | arxiv | Que hace | Distancia a nosotros |
|---|---|---|---|
| **AMROD** | `2406.16439` | Adaptacion continua de detector en stream de cámara (online TTA) | **El vecino mas cercano al caso de uso.** Adapta en cámara; baseline obligatorio. Diferencia nuestra: la fuente de senal (world-model temporal) y el foco en open-world unknowns, no solo dominio. |
| **OWLv2 / OWL-ST** | `2306.09683` | Self-training a escala para OVD (escala de datos web) | Self-training masivo; sin eje de estabilidad de largo horizonte en un stream unico. |
| **Scene-specific detector** (clasico) | `1611.07544` | Especializar un detector a una cámara fija via pseudo-labels (pre-foundation-models) | Precedente conceptual de hace ~10 anos: la idea de "especializar a la escena" no es nueva; lo nuevo (si algo) es la senal y la medicion de drift con foundation models. |

**Implicacion:** "analitica de personas en mall con cámara fija" es **commodity**. El mall
queda como **vehiculo de demo**, fuera del scope experimental del paper.

---

## 5. Bloque D — La batalla de backbones (lo que decide el gate)

Aca esta el verdadero campo de batalla. Nuestra tesis vive o muere segun que backbone
provea la mejor **senal de novedad open-world** a paridad de costo.

| Backbone | arxiv | Fortaleza | Por que importa |
|---|---|---|---|
| **V-JEPA 2.1** | `2603.14482` **(verificado 2026-06-17)** | Semantica de movimiento, error de **prediccion temporal** como senal de "sorpresa" | Nuestra apuesta: la senal temporal detecta eventos/objetos novedosos que un backbone estatico no "ve raros". |
| **DINOv3** | `2508.10104` **(verificado 2026-06-17)** | SOTA en **dense features** estaticas (Gram anchoring anti-degradacion en entrenamiento largo) | **La amenaza #1.** Ya gana en features densas estaticas. Si tambien gana en la senal de novedad, V-JEPA no aporta nada y la tesis cae. |

**Por que esto es el gate (Milestone 0):** la pregunta de investigacion
([`01_vision_y_plan.md`](01_vision_y_plan.md) §3) es falsable precisamente porque
DINOv3 puede ganar. El experimento C1 (DINOv3) vs C2 (V-JEPA 2.1) vs C3 (hibrido) a
idéntico costo es lo que separa "intuicion bonita" de "resultado defendible".

---

## 6. Bloque E — El hueco defendible (sintesis)

Cruzando los cinco bloques, casi todo esta ocupado. Lo que **no** encontramos publicado:

1. **La senal de novedad open-world derivada del error de prediccion temporal de un
   world-model de video**, evaluada cabeza-a-cabeza contra un backbone estatico fuerte
   (DINOv3) **a paridad de costo**, en stream de cámara fija. Los trabajos de self-training
   (Bloque B) usan uncertainty del detector o del VLM, no una senal temporal/predictiva.
2. **Un benchmark/protocolo de estabilidad de largo horizonte (drift a 1/3/7 dias)** para
   auto-entrenamiento con foundation models en cámara fija, con una metrica escalar tipo
   *Long-Horizon Stability Score*. AMROD adapta pero no instrumenta el colapso a dias;
   los demas reportan una pasada, no la curva temporal.
3. **El VLM como circuit-breaker externo al EMA disparado por la senal de novedad** (no
   en cada frame), como mecanismo anti-drift medible — frente a co-teaching (`2511.09955`)
   que ataca el mismo problema por otra via.

Estos tres huecos son **estrechos y falsables**, no "un sistema nuevo". Esa es justamente
la forma que un revisor de workshop acepta. La traduccion a hipotesis (H1/H2/H3),
criterios de exito/fracaso y venues esta en [`04_veredicto_novedad.md`](04_veredicto_novedad.md).

---

## 7. Matriz de "que nos diferencia" (resumen de una pagina)

| Eje | Lo que ya existe | Lo nuestro (si el gate pasa) |
|---|---|---|
| Fuente de la senal | uncertainty del detector / VLM (SAS-Det, DST-Det, VLM-PL) | **error de prediccion temporal de V-JEPA 2.1** |
| Comparacion de backbone | rara vez a paridad de costo | **C1/C2/C3 normalizados por FLOPs/latencia** |
| Anti-drift | calibracion de confianza, co-teaching | **VLM circuit-breaker externo al EMA disparado por novedad** |
| Horizonte | una pasada / online genérico | **curva 1/3/7 dias + Long-Horizon Stability Score** |
| Caso de uso | mall analytics (commodity) | mall solo como demo; el aporte es la senal + el benchmark |
| Integracion (lazo, SAM2, NL→detector) | **resuelto** (Autodistill, SAM2Auto) | **lo usamos, no lo reclamamos** |

---

## 8. Cola de lectura priorizada para colaboradores

Antes de escribir una linea de paper, leer en este orden (los que mas pueden invalidar
el hueco, primero):

1. **"The Detector Teaches Itself"** `2605.03642` — el riesgo directo: ya hace self-teaching detector+VLM. **Resuelto al leer el abstract:** opera sobre imagenes estaticas (COCO/LVIS) afinando el backbone del VLM; NO toca senal temporal, stream de cámara fija ni drift de largo horizonte. Cubre el patron, no nuestro hueco. Releer el PDF antes de afirmar la frontera.
2. **SAM2Auto** `2506.07850` — el vecino arquitectonico; delimitar la frontera "integracion vs senal".
3. **DINOv3** `2508.10104` — el rival del gate; entender donde gana y por que (dense features estaticas SOTA).
4. **V-JEPA 2.1** `2603.14482` — confirmar que la senal de prediccion temporal es extraible del encoder congelado (el abstract confirma Dense Predictive Loss + self-supervision jerarquica).
5. **AMROD** `2406.16439` — el baseline de adaptacion en cámara (es test-time adaptation continuo).
6. **SAS-Det** `2308.06412` ("Taming Self-Training for OVD") y **DST-Det** `2310.01393` — los baselines de self-training OVD.
7. **Co-teaching VLM** `2511.09955` — la alternativa anti-drift a comparar (per-object co-teaching sobre KITTI/ACDC/BDD, estatico).

> Regla de la honestidad (de [`CLAUDE.md`](../CLAUDE.md)): existencia del ID verificada
> NO es lo mismo que claim verificado. Los 18 IDs existen (chequeo 2026-06-17), pero antes
> de afirmar un resultado especifico en texto publicable hay que abrir el PDF. La novedad
> se sostiene sobre hechos verificables; el resto son lineas de investigacion, no resultados.
