# Competidores y componentes externos

Síntesis de la vigilancia tecnológica de junio 2026. Agrupa lo encontrado por
**rol respecto a nuestros dos pipelines** y deja todos los links al final.

> Honestidad: lo marcado **(verificado por fetch 2026-06-17)** se confirmó
> abriendo la página. El resto está **a leer**: el ID/URL existe pero no validé
> el claim. Antes de citar cualquier número, abrir el PDF.

---

## 1. El stack que absorbe el Pipeline 1 (detectar + segmentar + seguir "X")

Esto es lo que hace que detectar/segmentar/seguir un objeto genérico ya **no
sea** un aporte: hay varios productos que lo resuelven de punta a punta.

### SAM 3 / SAM 3.1 — "Segment Anything with Concepts" (arxiv 2511.16719)
- **Lo importante:** unifica **detección + segmentación + tracking** a partir de
  un *prompt de concepto*: una frase nominal ("persona con carrito"), un
  **exemplar de imagen**, o ambos. Es exactamente el prompt "texto + 1 ejemplo"
  que el proyecto quería.
- **SAM 3.1** agrega video en tiempo real. Pesos liberados; repo
  `facebookresearch/sam3`.
- **Implicación:** colapsa todo el Pipeline 1 en un solo modelo. Es el candidato
  a front-end de P1, reemplazando la cadena detector-open-vocab + SAM2 separados.

### SAM2-OV — Open-Vocabulary Multi-Object Tracking (AAAI) — (verificado por fetch 2026-06-17)
- **Lo importante:** tracking multi-objeto open-vocab sobre SAM2.
- **Lo que NO hace (confirmado leyendo la página):** **no** razona sobre
  comportamiento, **no** hace pseudo-labeling, **no** tiene lazo de
  auto-corrección. Es tracking y se acaba ahí.
- **Implicación:** absorbe el Pipeline 1, **nada** del Pipeline 2 ni del lazo.

### Grounded-SAM-2 (IDEA-Research)
- Grounding DINO (cajas desde texto) + SAM2 (máscaras + propagación). Pipeline 1
  clásico ya empaquetado. Absorbe P1.

### Variantes de tracking open-vocab (a leer)
- **SAM2MOT** (arxiv 2504.04519) — MOT sobre SAM2.
- **SLAck: Open-Vocabulary Tracking** (arxiv 2409.11235) — asociación
  semántica + localización + apariencia para tracking OV.
- **OpenWorldSAM** (arxiv 2507.05427) — segmentación open-world sobre SAM.
- Todas caen en "absorben P1, no tocan P2".

---

## 2. El stack métrico (tracking + conteo SIN semántica)

Lo que el usuario identificó como "un SOTA para métricas en el tiempo haciendo
trackeo pero sin semántica". Esto cubre el caso **métrico** (contar, tasa,
densidad, velocidad, dwell) de forma barata, determinista y en tiempo real —
**sin VLM**. No pongas un VLM a contar.

- **Ultralytics YOLO** — detección + tracking (BoT-SORT / ByteTrack) + conteo de
  objetos out-of-the-box. Es el camino corto para todo lo métrico.
- **Métricas MOT** — HOTA / MOTA / IDF1; "Local Metrics for Multi-Object
  Tracking" (arxiv 2104.02631) para evaluación fina.
- **CVAT — Top AI Models for Object Tracking** — panorama de modelos de tracking.

**Distinción que ordena el diseño (metric vs semantic):**
- **Métrico** (contar latas, regular velocidad de línea): detector + tracker +
  aritmética. Determinista, barato, tiempo real. **No necesita VLM.**
- **Semántico** (merodeo, pelea, aglomeración, anomalía): necesita la señal de
  V-JEPA + interpretación VQA. Caro, on-demand. **Aquí entra el aporte.**

El caso "latas en línea de producción" es casi todo métrico; el caso "masas
humanas" es semántico.

---

## 3. La pieza temporal nuestra (Pipeline 2)

- **V-JEPA 2** (arxiv 2506.09985) — encoder de video auto-supervisado. **No es
  un detector**: da embeddings + una señal de novedad/"sorpresa" (un *dónde/
  cuándo*, no un *qué*). Es la base del Pipeline 2.
- **V-JEPA2 + YOLO para video en tiempo real** (Medium) — ejemplo informal de
  combinar el encoder temporal con un detector YOLO; útil como referencia de
  plumbing, no como fuente académica.

Recordatorio: V-JEPA solo se justifica por el **eje temporal**. Para apariencia
estática, DINOv3 (arxiv 2508.10104) es SOTA y lo dejaría redundante — por eso el
GATE compara las dos.

---

## 4. VQA / razonamiento de comportamiento (las dos "gorras" del VLM)

- **Gorra A — verificador** (rápido, grounded, por detección): Florence-2 /
  PaliGemma / Qwen2-VL 2B.
- **Gorra B — razonador de comportamiento** (video, on-demand): Qwen2-VL 7B /
  InternVL / LLaVA-Video.
- **Common ground recomendado:** familia **Qwen2-VL (2B + 7B)** para cubrir
  ambas gorras con un solo ecosistema.

El VQA entra en 3 puntos, no es una caja del pipeline: (1) parseo del prompt,
(2) interpretación de comportamiento (el VQA core = gorra B), (3) reporte/consulta.

---

## Links (consolidados)

### Tracking / métricas (commodity, Pipeline 1 + métrico)
- Top AI Models for Object Tracking — CVAT: https://www.cvat.ai/resources/blog/top-ai-models-video-tracking
- Multi-Object Tracking with Ultralytics YOLO: https://docs.ultralytics.com/modes/track
- Object Counting using Ultralytics YOLO: https://docs.ultralytics.com/guides/object-counting
- Local Metrics for Multi-Object Tracking (arxiv 2104.02631): https://arxiv.org/pdf/2104.02631

### Detección/segmentación/tracking open-vocab (absorben Pipeline 1)
- SAM2-OV: Open-Vocabulary Multi-Object Tracking (AAAI): https://ojs.aaai.org/index.php/AAAI/article/view/37301
- Grounded-SAM-2 (IDEA-Research, GitHub): https://github.com/IDEA-Research/Grounded-SAM-2
- SAM2MOT (arxiv 2504.04519): https://arxiv.org/pdf/2504.04519
- SLAck: Open-Vocabulary Tracking (arxiv 2409.11235): https://arxiv.org/pdf/2409.11235
- OpenWorldSAM (arxiv 2507.05427): https://arxiv.org/html/2507.05427v2

### SAM 3 / 3.1 (front-end candidato de Pipeline 1)
- SAM 3: Segment Anything with Concepts (arxiv 2511.16719): https://arxiv.org/abs/2511.16719
- SAM 3 — Meta AI Research: https://ai.meta.com/research/sam3/
- SAM 3 / 3.1 — Meta AI Blog: https://ai.meta.com/blog/segment-anything-model-3/
- facebookresearch/sam3 (GitHub): https://github.com/facebookresearch/sam3
- Introducing Meta SAM 3 (YouTube): https://www.youtube.com/watch?v=G4OLPDjwncw
- Segment Anything Playground (demo Meta): https://aidemos.meta.com/segment-anything/

### V-JEPA (pieza temporal, Pipeline 2)
- V-JEPA2 + YOLO for Real-Time Video Understanding (Medium): https://medium.com/@soumyajit.swain/v-jepa2-and-yolo-combining-vision-and-action-for-real-time-video-understanding-762baf50610b
- Introducing V-JEPA 2 — Meta AI: https://ai.meta.com/vjepa/

### Opcionales (surgieron en búsqueda, a leer)
- Leveraging VLMs for Open-Vocabulary Instance Segmentation and Tracking (arxiv 2503.16538): https://arxiv.org/abs/2503.16538
- CloudTrack (arxiv 2409.16111): https://arxiv.org/abs/2409.16111
