# 01 - Visión y plan de trabajo

> Documento de onboarding para colaboradores que se suman al proyecto.
> Honesto y accionable, no marketing. Si una frase suena a hype, está mal escrita.
> Fecha base: 2026-06-17. Lectura obligatoria antes de tocar código:
> [`CLAUDE.md`](../CLAUDE.md), [`02_arquitectura.md`](02_arquitectura.md),
> [`04_veredicto_novedad.md`](04_veredicto_novedad.md) y [`papers.md`](../papers.md).

---

## 1. TL;DR

Construimos un detector de objetos sobre **cámara fija** que se auto-corrige con
pseudo-labels verificados por un VLM, usando V-JEPA 2.1 (un *world-model* de video)
como aporte central. Lo que **afirmamos como contribución** NO es el lazo ni el
patrón "VLM verifica pseudo-labels" (eso ya está publicado, ver §2): es que el
**error de predicción de V-JEPA** provee una señal temporal de *sorpresa* útil para
**detección de novedad open-world** y para **estabilidad de largo horizonte** en
streams continuos. **Regla de oro: primero el gate.** No construimos el sistema
completo hasta correr un experimento decisivo de 2-3 semanas (V-JEPA vs DINOv3 vs
híbrido). Si V-JEPA no gana a paridad de costo, pivotamos o paramos —
[ver el veredicto completo](04_veredicto_novedad.md).

---

## 2. Idea central

La visión completa es seductora: una cámara en un mall arranca con una idea básica
en lenguaje natural de qué detectar ("una persona", "alguien con un carrito") y se
auto-corrige hasta detectar casi sola, sin re-etiquetado humano constante. Hay un
lazo que combina detección open-vocabulary (Grounding DINO / YOLO-World),
segmentación temporal (SAM2), tracking, un encoder V-JEPA congelado, un VLM que
verifica, y guardrails anti confirmation-bias. Toda esa máquina está descrita en
[`02_arquitectura.md`](02_arquitectura.md) y vale como **infraestructura**.

Ahora la parte honesta, y es la más importante que un colaborador tiene que
internalizar: **casi todo eso es andamiaje, no aporte.** El veredicto del juez
([`04_veredicto_novedad.md`](04_veredicto_novedad.md)) es explícito: ante un revisor,
el sistema "tal como está planteado" es *integración de APIs ya publicadas*. El
patrón "VLM verifica pseudo-labels para auto-entrenar un detector" está publicado
5+ veces (Autodistill, VLM-PL `2403.05346`, SAS-Det `2308.06412`,
DST-Det `2310.01393`). "NL → destila un detector" es literalmente el producto
Autodistill. "Analítica de personas en mall" es commodity.

**Lo único que sobrevive a un revisor** es la señal temporal/predictiva del
world-model como motor de open-world en streams continuos: el VLM actúa como
**circuit-breaker externo al EMA** para frenar el drift. Eso —y solo eso— es la
tesis. El mall, la demo Streamlit, la destilación y el split LOCAL/COLAB se
**conservan como vehículo de demo e infraestructura experimental**, nunca como el
aporte. La frase "intuición a nivel humano / mejora casi solo" se **borra de todo
documento técnico**: son red flags para revisores.

---

## 3. Problemática: pregunta de investigación

Una sola pregunta, falsable (puede perder):

> ¿Puede el error de predicción de un world-model de video (V-JEPA 2.1) servir como
> señal de novedad open-world que (i) supere a backbones estáticos (DINOv3) y a
> self-training puro en **recall de unknowns**, y (ii) frene el **drift** de
> auto-entrenamiento en horizonte largo (1/3/7 días) en cámara fija, **a idéntico
> presupuesto de cómputo y supervisión**?

Es falsable porque si DINOv3 iguala o supera, la tesis cae. Esa propiedad es
exactamente lo que la hace publicable.

---

## 4. Hipótesis

Formuladas para ser testeables (los símbolos como X se fijan en el gate, §7):

- **H1 — temporal > estático (principal).** En recall de unknowns, la señal de
  sorpresa de V-JEPA 2.1 supera a la misma señal derivada de DINOv3 por un margen
  ≥ X. *Si H1 falla, el world-model no aporta y el proyecto pierde su razón de ser.*
- **H2 — circuit-breaker anti-drift.** Un VLM externo al EMA, disparado por la señal
  de novedad, reduce la degradación de mAP a 7 días respecto de self-training puro,
  evitando el colapso (definición operacional de colapso en §7).
- **H3 — no-redundancia.** La señal de sorpresa temporal es complementaria, no
  redundante, con la incertidumbre del detector (correlación baja + ganancia aditiva
  en ablation). Esta es la que convierte "otra señal de uncertainty" en "señal nueva".

---

## 5. Objetivos

**Objetivo general.** Determinar, con evidencia cuantitativa y reproducible, si el
error de predicción de V-JEPA 2.1 es una señal de novedad open-world defendiblemente
superior a alternativas estáticas, y construir alrededor de ella un lazo de
auto-corrección cuya estabilidad de largo horizonte sea medible.

**Objetivos específicos (medibles):**

1. **Pasar el GATE (Milestone 0, §7) — bloqueante.** Correr V-JEPA vs DINOv3 vs
   híbrido a paridad de costo y decidir GO / NO-GO. *Ningún otro objetivo arranca en
   serio antes de este.*
2. Implementar el cálculo de la **señal de novedad** desde el encoder congelado en
   las 3 condiciones (C1/C2/C3) sobre el mismo pipeline.
3. Convertir el `DriftMonitor` en un **instrumento de medición de drift** que produzca
   la curva de degradación a 1/3/7 días (mAP knowns + recall unknowns + tasa de FP).
4. Validar **H2**: cuantificar la reducción de degradación de mAP a 7 días con el VLM
   circuit-breaker vs self-training puro, con ablations (con/sin VLM, con/sin señal).
5. Validar **H3**: medir correlación señal-temporal vs uncertainty del detector y la
   ganancia aditiva al combinarlas.
6. Producir un **protocolo de evaluación de drift** reproducible (split temporal,
   unknowns anotados) y una métrica escalar tipo *Long-Horizon Stability Score* —
   un benchmark de drift en cámara fija con foundation models no existe hoy.

---

## 6. Arquitectura propuesta y cómo se une todo

El sistema tiene dos lados con responsabilidades estrictamente separadas. En **LOCAL**
(Windows, sin GPU de entrenamiento) corre solo la web demo y la inferencia liviana del
modelo destilado. En **COLAB** (GPU 80GB) corre todo el heavy lifting: percepción
pesada, VLM, lazo de pseudo-labeling y entrenamiento/destilación. El contrato que los
une es unidireccional: Colab produce `distilled_detector.pt` y ese artefacto baja a
`models/`. El switch lo decide `core/config.py` vía `AMTA_RUNTIME` y `AMTA_TIER`.

El flujo por frame en Colab encadena: detector open-vocab propone cajas desde prompts
→ SAM2 refina a máscaras con memoria temporal → Tracker asigna `track_id` →
V-JEPA encoder (congelado) aporta embeddings de semántica/novedad/re-ID → el VLM
verifica y corrige → solo lo verificado y de score alto se vuelve pseudo-label → el
`DriftMonitor` vigila confirmation-bias y samplea para revisión humana → periódicamente
el `Trainer` afina y destila. Todos los umbrales viven centralizados en
`core/config.py::LoopThresholds` y son inviolables.

**No redibujamos nada acá.** Los 7 diagramas viven en
[`02_arquitectura.md`](02_arquitectura.md) y son la referencia canónica:

| # | Diagrama | Para qué mirarlo |
|---|---|---|
| 1 | Vista de sistema: topología LOCAL vs COLAB | entender dónde corre cada cosa y el contrato del checkpoint |
| 2 | Pipeline de percepción por frame | el camino de un frame: detector → SAM2 → tracker → encoder |
| 3 | El lazo de auto-corrección (ciclo central) | cómo se generan y filtran los pseudo-labels |
| 4 | Secuencia de una iteración del lazo | el orden exacto de llamadas en `_process_full_loop` |
| 5 | Ciclo de vida de entrenamiento, destilación y promoción | las 4 fases bootstrap → auto-etiquetado → train → promoción |
| 6 | Máquina de estados de una detección | el ciclo de vida de una sola `Detection` y sus cortes |
| 7 | Mapa de guardrails y umbrales de `LoopThresholds` | qué umbral controla qué punto del lazo |

> Recordá el reencuadre del §2: en el **paper**, SAM2/segmentación, destilación, demo
> y mall salen del scope experimental. En el **repo**, siguen existiendo como
> infraestructura. No los borres del código; sí del relato de la contribución.

---

## 7. El GATE (Milestone 0)

Este es el experimento que decide TODO. Cuesta 2-3 semanas y puede ahorrar 6 meses.
**Se hace ANTES de tocar el resto del sistema.**

**Setup mínimo.** 1 cámara fija (mall o cualquier stream continuo público de horas),
el lazo ya implementado lo justo para correr, y la señal de novedad calculada de tres
formas sobre el **mismo** pipeline y el **mismo** presupuesto de cómputo/supervisión:

- **C1 — DINOv3** como fuente de la señal (backbone estático, baseline fuerte).
- **C2 — V-JEPA 2.1** error de predicción como señal de sorpresa temporal.
- **C3 — Híbrido** (DINOv3 espacial + V-JEPA temporal).

**Qué se mide.**

| Métrica | Definición |
|---|---|
| **Recall de unknowns** | objetos/eventos novedosos recuperados vs ground-truth manual (unknowns inyectados o anotados). |
| **AUROC de la señal** | separabilidad known/unknown usando la señal como score. |
| **Complementariedad** | correlación de Pearson entre señal V-JEPA y uncertainty del detector (buscamos baja); y Δrecall al sumarlas. |
| **Costo** | latencia/FLOPs por frame de cada condición, para normalizar la comparación. |

**Criterio de ÉXITO (continuar el pivote):**

- C2 o C3 supera a C1 (DINOv3) en recall de unknowns por **≥ 5 puntos absolutos**
  (o AUROC **≥ +0.05**), **a paridad o menor costo**; **y**
- correlación señal-temporal vs uncertainty del detector **< 0.5** (complementaria); **y**
- reproducible en **≥ 2 streams** distintos (no overfit a una cámara).

**Criterio de FRACASO (matar el framing V-JEPA):**

- C1 (DINOv3) iguala o supera a C2/C3 dentro de **±2 puntos**; **o**
- la ventaja de V-JEPA desaparece al normalizar por costo; **o**
- la señal temporal correlaciona **> 0.7** con la uncertainty existente (redundante).

**Qué decidimos según el resultado:**

- **GATE OK →** seguimos el plan completo, apuntando a workshop top-tier
  (probabilidad honesta 40-55%). Recién acá invertimos en ablations, baselines y el
  benchmark de drift.
- **GATE FALLA →** NO-GO al framing world-model. Dos salidas honestas: (a) reescribir
  el paper sin V-JEPA, como estudio de **estabilidad de largo horizonte de
  self-training + VLM circuit-breaker** (más chico, sin mall, todavía publicable en
  workshop si el anti-drift está bien medido); o (b) parar. No insistir con
  "world-model + mall".

**Definición operacional de "colapso"** (para H2 y la curva de drift): caída > 10% de
mAP respecto del pico, o crecimiento monótono de FP por 3 ventanas seguidas.

---

## 8. Plan de trabajo y delegación

Tres roles. El objetivo del split es que el gate (Milestone 0) lo ataquen **A y B en
paralelo**: A construye las condiciones de señal (C1/C2/C3), B construye el lazo y la
instrumentación de medición que las evalúa.

| Rol | Responsabilidades | Primeros 3 entregables | Dueño de (archivos del repo) |
|---|---|---|---|
| **Líder (usuario)** | Dirección científica, criterio del gate, framing del paper, decisión GO/NO-GO, contrato LOCAL↔COLAB, integración. Custodio de `LoopThresholds` y de la honestidad del relato (corta hype). | (1) Congelar el protocolo del gate y el set de validación con unknowns. (2) Definir el *Long-Horizon Stability Score*. (3) Wiring de `orchestrator.py` para correr C1/C2/C3 a paridad de costo. | `core/orchestrator.py`, `core/config.py`, `core/types.py`, `colab/train_pipeline.ipynb`, `docs/` |
| **Colaborador A — Percepción / backbones** | La pieza que decide el gate: implementar la **señal de novedad** desde encoders congelados en las 3 condiciones (DINOv3, V-JEPA 2.1, híbrido), medir AUROC/recall/costo. SAM2 y el detector open-vocab como soporte. | (1) `VJepa2Encoder.extract_features` real + cálculo del error de predicción como score de sorpresa (entrypoints `vjepa2_1_*`). (2) Condición C1 (DINOv3) y C3 (híbrido) sobre el mismo pipeline, normalizadas por costo. (3) Reporte AUROC + recall de unknowns + FLOPs/frame por condición en ≥2 streams. | `core/perception/encoder.py`, `core/perception/detector.py`, `core/perception/segmenter.py`, `core/perception/tracker.py` |
| **Colaborador B — Lazo de auto-corrección** | El circuit-breaker y la instrumentación: VLM verificador disparado por novedad, pseudo-labeling, y convertir el `DriftMonitor` en medidor de drift de largo horizonte (curvas 1/3/7 días, métricas, benchmark). | (1) `VlmVerifier.verify_batch` real, externo al EMA, con trigger por señal de novedad (vs siempre). (2) `PseudoLabeler.emit` + `is_trustworthy` respetando `LoopThresholds`. (3) `DriftMonitor` que emite la curva de degradación + Long-Horizon Stability Score y el split temporal del benchmark. | `core/reasoning/vlm_verifier.py`, `core/loop/pseudo_labeler.py`, `core/loop/drift_monitor.py`, `core/loop/trainer.py` |

> Regla transversal: nadie hardcodea umbrales; todo desde `core/config.py`. Nadie carga
> V-JEPA/SAM2/VLM ni entrena en la ruta LOCAL — eso es Colab.

---

## 9. Delegación de investigación

La síntesis completa del prior-art (qué ya existe, qué tan cerca está y el hueco
defendible) vive en [`03_estado_del_arte.md`](03_estado_del_arte.md), con la cola de
lectura priorizada. Cada colaborador es dueño de un cuerpo de literatura. IDs de arxiv
para arrancar:

**Colaborador A (percepción / backbones / señal):**

- V-JEPA 2 (`2506.09985`) y V-JEPA 2.1 (`2603.14482`, *verificado 2026-06-17*) — el
  encoder y por qué 2.1 mejora dense features (Dense Predictive Loss + self-supervision
  jerárquica).
- **DINOv3** (`2508.10104`, *verificado 2026-06-17*) — el baseline estático a vencer en
  el gate (ya gana en dense features estáticas; ese es justo el riesgo de H1).
- SAM 2 (`2408.00714`), Grounding DINO (`2303.05499`), YOLO-World (`2401.17270`),
  Mask2Former (`2112.01527`) — soporte de percepción.

**Colaborador B (lazo / self-training / drift):**

- Baselines obligatorios del paper: SAS-Det (`2308.06412`), DST-Det (`2310.01393`),
  "The Detector Teaches Itself" (`2605.03642`), co-teaching VLM (`2511.09955`),
  VLM-PL (`2403.05346`), Autodistill (NL→destila).
- AMROD (`2406.16439`) — el vecino más cercano en adaptación de detector en cámara.
- SAM2Auto (`2506.07850`) — orquestación SAM2+DINO+VLM (para diferenciarse: eso es
  integración, no nuestra tesis).

> Validación de fuentes (2026-06-17): los 18 IDs de arxiv del proyecto fueron
> verificados contra arxiv.org y **todos son reales** (detalle en
> [`03_estado_del_arte.md`](03_estado_del_arte.md)). Se corrigió un error: YOLO-World
> era `2401.17270`, no `2405.14874`. Salvedad: existencia del ID ≠ claim verificado;
> antes de citar un resultado específico, abrir el PDF.

---

## 10. Riesgos y cómo los mitigamos

- **DINOv3 ≥ V-JEPA (riesgo #1, mata la tesis).** DINOv3 ya gana en localización
  estática; si gana también en la señal de novedad, no queda contribución. *Mitigación:
  el gate (§7) lo expone en la semana 3 en vez del mes 5; si falla, pivotamos sin
  haber gastado 6 meses.*
- **"Es ingeniería, no investigación".** El lazo, los guardrails y la orquestación ya
  están publicados. *Mitigación: el relato del paper se centra 100% en la señal
  temporal y el anti-drift medible; mall, demo, destilación y SAM2 salen del scope
  experimental (quedan como infraestructura).*
- **Drift / confirmation-bias.** El lazo puede reforzar sus propios errores.
  *Mitigación: VLM circuit-breaker externo al EMA, guardrails de `LoopThresholds`,
  muestreo human-in-the-loop, y el `DriftMonitor` como instrumento que mide el colapso
  en vez de esconderlo. Mostrar el colapso sin guardrails es parte del aporte.*

---

## 11. Glosario rápido

- **V-JEPA** — Video Joint-Embedding Predictive Architecture. Encoder de video
  auto-supervisado de Meta FAIR; predice en espacio de representación latente, no en
  píxeles. Da **embeddings**, no cajas ni máscaras. Usamos **V-JEPA 2.1**.
- **JEPA** — el paradigma general: aprender prediciendo representaciones latentes de
  partes ocultas de la entrada, en lugar de reconstruir píxeles.
- **OVD (Open-Vocabulary Detection)** — detección que acepta clases nuevas vía texto
  en lenguaje natural, sin re-entrenar (Grounding DINO, YOLO-World). Permite arrancar
  con "una idea básica".
- **SAM2** — Segment Anything Model 2: convierte cajas en máscaras y las **propaga en
  el tiempo** con memoria temporal.
- **VLM-as-critic** — usar un Vision-Language Model (Florence-2, InternVL) como
  **verificador externo** ("¿esto es realmente X?"). Los VLM son malos
  auto-corrigiéndose pero buenos verificando.
- **Pseudo-label** — etiqueta generada por el propio sistema (no por un humano) que,
  si pasa los umbrales y la verificación, se usa para reentrenar.
- **Drift** — degradación progresiva del modelo cuando se auto-entrena con sus propios
  pseudo-labels (acumula y refuerza errores).
- **Open-world** — el detector debe lidiar con objetos/eventos **no vistos** en
  entrenamiento (*unknowns*), no solo con un conjunto cerrado de clases.
- **Dense features** — representaciones por-píxel/por-región (no un solo vector por
  imagen), necesarias para localización y segmentación. V-JEPA 2.1 las mejora.
- **Circuit-breaker** — mecanismo que corta el lazo de auto-entrenamiento cuando
  detecta riesgo de drift; en nuestro caso, el VLM **externo al EMA** disparado por la
  señal de novedad.
- **EMA (Exponential Moving Average)** — promedio temporal de pesos usado como
  "teacher" en self-training. Mantener el circuit-breaker **fuera** del EMA evita
  realimentar el error.
