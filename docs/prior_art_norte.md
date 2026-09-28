# Prior art del norte — trabajos similares (2026-09-28)

> Relevamiento con Firecrawl de trabajos parecidos a `docs/norte.md` (prompt → detección →
> tracking → eventos → series temporales para EDA/ML): empresas grandes, academia, open
> source y contribuidores individuales.
>
> Convención de verificación: **✓** = página oficial abierta y leída de primera mano ·
> **✓(abs)** = solo título + abstract leídos (abrir el PDF antes de citar un resultado
> concreto) · **⚠️** = solo aparece en resultados de búsqueda, no leído.

## 1. Lectura en una línea

La **capa de modelo** (segmentar y seguir "cualquier cosa" desde texto) ya es commodity y
abierta (Meta SAM 3/3.1). Las **primitivas de conteo** (línea, zona, permanencia) también
(Ultralytics, Roboflow supervision). Lo que las big tech ofrecían como servicio administrado
para convertir eso en datos (Google, AWS, Microsoft) **se está retirando**. La única
plataforma completa y activa es **NVIDIA VSS**, pesada y atada a su stack. El hueco que
apunta el norte, una capa liviana de **prompt → eventos → series con contrato de datos
para EDA/ML y costo medido**, existe. Pero la idea en sí **no es nueva**: el aporte hay que
demostrarlo, no suponerlo.

## 2. Meta — la capa de modelo

| Trabajo | Qué es | Relevancia para el norte | Verif. |
|---|---|---|---|
| **SAM 3** (nov-2025) — [ai.meta.com/research/sam3](https://ai.meta.com/research/sam3/), [arXiv 2511.16719](https://arxiv.org/abs/2511.16719) | "Promptable concept segmentation": con un texto corto o un ejemplo (caja), **detecta, segmenta y sigue todas las instancias** del concepto en imágenes y video | Es el "algo como lo que hace SAM" del norte. Candidato directo para N2 (reemplaza detector + SAM2 por un solo modelo) | ✓ |
| **SAM 3.1** (27-mar-2026) — [blog](https://ai.meta.com/blog/segment-anything-model-3/), [HF facebook/sam3.1](https://huggingface.co/facebook/sam3.1) | Reemplazo drop-in con "object multiplexing" (hasta 16 objetos por forward pass) | Duplica el throughput en video: 16 → 32 fps en H100 con una cantidad media de objetos | ✓ |
| **SAM 3 Agent** (mismo release) | Un MLLM usa SAM 3 como herramienta: propone frases, revisa máscaras e itera | Patrón para prompts complejos ("personas con carrito que entran a la tienda A") sin entrenar | ✓ |
| **Perception Encoder / PLM** (abr-2025) — [github](https://github.com/facebookresearch/perception_models) | Encoders de imagen/video + modelo de lenguaje perceptual | Son los encoders de texto e imagen **dentro** de SAM 3 | ✓ |
| Integración **Roboflow × SAM 3** — [blog](https://blog.roboflow.com/sam3/) | SAM 3 en Workflows, Inference local o en nube privada, y fine-tuning | Vía rápida para prototipar y para fine-tunear SAM 3 con pocos datos | ✓ |

**Datos duros que condicionan la adopción (leídos en las fuentes oficiales):**
- Licencia **"SAM License"** (no Apache/MIT): permite uso, redistribución y derivados de
  forma royalty-free, pero prohíbe usos militares/ITAR, de espionaje y bajo sanciones, y
  tiene cláusula de terminación por litigio de patentes. Revisarla antes de un uso comercial.
- Requisitos: Python ≥ 3.12, PyTorch ≥ 2.7, **CUDA ≥ 12.6**.
- **Memoria: ~17,7 GB con 5 objetos en H100** (SAM 3.1 compilado, según el commit de
  release). **Supera los 15 GB de la T4** de Colab y los 6 GB de la RTX 2060 de Alex.
  Hay que medirlo en nuestro hardware; es el riesgo principal para adoptarlo.
- En video, el costo **escala linealmente con el número de objetos**: SAM 3 era "casi tiempo
  real con ~5 objetos concurrentes" en H200. Para escenas concurridas (una tienda llena)
  es una restricción real.
- Solo acepta frases nominales cortas ("a hardcover book"), no descripciones largas; para
  eso está SAM 3 Agent + MLLM. Generaliza mal a conceptos finos fuera de dominio sin
  fine-tuning.
- El repo sigue activo: el 2026-09-18 se agregó un modo streaming que libera las máscaras de
  frames ya procesados, pensado para grabaciones largas. Es justo nuestro caso de cámara
  continua.

## 3. Grandes empresas — plataformas de analítica de video

| Plataforma | Qué hace | Estado | Verif. |
|---|---|---|---|
| **NVIDIA VSS 3.2** (Video Search & Summarization) — [docs](https://docs.nvidia.com/vss/latest/) | Arquitectura de referencia en 3 capas: (1) RT-CV con DeepStream + RT-DETR / **Grounding DINO** + tracking, RT-Embedding y RT-VLM; (2) **Behavior Analytics**: consume metadatos por Kafka/Redis/MQTT, sigue objetos en el tiempo, calcula velocidad/dirección/trayectoria y detecta **tripwire crossings y ROI entry/exit**; alertas verificadas con VLM y guardadas en Elasticsearch; (3) agente con MCP para reportes y preguntas | **Activo.** Es el análogo más completo del norte. Pesado: microservicios, NIMs, GPU NVIDIA | ✓ |
| **NVIDIA deepstream-occupancy-analytics** — [github](https://github.com/NVIDIA-AI-IOT/deepstream-occupancy-analytics) | Ejemplo: cuenta personas que cruzan un tripwire (NvDsAnalytics) y manda los datos en vivo por Kafka | Ejemplo mínimo, clases fijas (PeopleNet) | ✓ |
| **Google Vertex AI Vision — Occupancy analytics** — [docs](https://docs.cloud.google.com/vision-ai/docs/occupancy-analytics-model) | Cuenta personas/vehículos con **active zones, line crossing (con dirección) y dwell time**, con salida estructurada **a BigQuery** para analítica | **Deprecado el 15-jun-2026, End of Life el 30-sep-2026** | ✓ |
| **AWS Rekognition Streaming Video Events** — [página](https://aws.amazon.com/rekognition/video-features/) | Detecta personas/mascotas/paquetes en streams, con caja + timestamp, para alertas | **Cerrado a clientes nuevos desde el 30-abr-2026.** Orientado a alertas del hogar, no a series | ✓ |
| **Microsoft Rocket** — [github](https://github.com/microsoft/Microsoft-Rocket-Video-Analytics-Platform) | Pipeline edge+cloud (C#/.NET) de conteo y alertas con **cascada de costo** | **Archivado el 11-jun-2026**; último commit en 2022 | ✓ |

Lectura: el triplete de Google (**zonas activas / cruce de línea / permanencia → BigQuery**)
y la Behavior Analytics de VSS (**tripwire / ROI entry-exit / trayectoria → broker → DB**)
**validan casi 1:1 la capa L2 de eventos de `norte.md`**. No hay que inventar la taxonomía de
eventos: hay que adoptarla.

## 4. Academia — sistemas de consulta sobre video

| Trabajo | Idea | Relación con el norte | Verif. |
|---|---|---|---|
| **NoScope** (Kang et al., 2017) — [arXiv 1703.02529](https://arxiv.org/abs/1703.02529) | Cascadas de modelos baratos → caros para abaratar consultas sobre video | Antecedente directo de la **cascada N1→N2 de Alex** | ✓(abs) |
| **Focus** (OSDI 2018) · **Chameleon** (SIGCOMM 2018, Microsoft) | Indexación barata al ingestar / adaptación de configuración para costo-precisión | Mismo eje costo-precisión que mide `core/cascade/cost/` | ⚠️ |
| **EVA / EvaDB** (Georgia Tech, VLDB 2023 demo) — [demo](https://kexinrong.github.io/papers/eva-demo-vldb23.pdf) | Base de datos relacional con IA: consultas tipo SQL sobre video | **Repo archivado el 15-oct-2025** | ✓(abs) |
| **Familia VOCAL** (UW Database Group) — [VisualWorld](https://db.cs.washington.edu/projects/visualworld/) | VOCAL (CIDR 2022), EQUI-VOCAL (PVLDB 2023: sintetiza consultas de eventos compuestos a partir de pocos ejemplos, con scene graphs espacio-temporales), VOCALExplore (PVLDB 2023: exploración y modelos pay-as-you-go), **VOCAL-UDF (SIGMOD 2025: un LLM construye los módulos que faltan como UDFs)** | Consultas **compuestas** de eventos ("persona entra y luego se queda >30 s") sobre tracks | ✓(abs) |
| **LAVA** (ACM MM 2025) — [arXiv 2507.19821](https://arxiv.org/abs/2507.19821), [code](https://github.com/yuyanrui/LAVA) | Analítica de tráfico **por lenguaje natural**: detección open-world + trayectorias largas; consultas de selección, **agregación** y top-k; benchmark propio | Lo más cercano en academia a "prompt → conteo agregado". Métrica útil para evaluar L3: **error de agregación (MPAE/MAPE)** | ✓(abs) |
| **AVA** (NSDI 2026) — [arXiv 2505.00254](https://arxiv.org/abs/2505.00254) | Analítica agéntica con VLM: **Event Knowledge Graphs** en casi tiempo real para video largo/continuo + recuperación agéntica | Alternativa "VLM-first" a nuestra ruta "detector-first" | ✓(abs) |
| **Concord** (arXiv, 4-sep-2026) — [arXiv 2609.05756](https://arxiv.org/abs/2609.05756) | Álgebra relacional sobre videos, frames y **object tracks**; reemplaza llamadas a MLLM por **detect-track-join** (F1 .364 → .813 sin llamadas a MLLM, en un clip pequeño) | Argumento de costo a favor de nuestra ruta: detección + tracks como tablas y el LLM solo donde hace falta | ✓(abs) |

## 5. Open source y contribuidores individuales

| Proyecto | Qué aporta | Verif. |
|---|---|---|
| **Ultralytics Solutions** — [docs](https://docs.ultralytics.com/solutions) | Object Counting, Counting in Regions, Queue Management, Heatmaps, Analytics, Speed, Parking… **Alex ya usa `ObjectCounter` y `QueueManager`** | ✓ |
| **Roboflow supervision + Workflows** — [dwell time](https://blog.roboflow.com/dwell-time-and-zone-analytics/), [supervision](https://supervision.roboflow.com/) | `LineZone`, `PolygonZone`, **Time in Zone** (permanencia por tracker ID + conteo único en vivo); distingue bien *foot traffic* (pasó), *people counting* (visitante único) y *dwell* (se detuvo) | ✓ |
| **Frigate NVR** (~36k ⭐) — [discusión #3676](https://github.com/blakeblackshear/frigate/discussions/3676) | NVR open source con detección y zonas. En 2022 la comunidad pidió una **"capa de analítica de eventos"** (frecuencia por cámara, zona, tipo de objeto y hora del día): señal de demanda de lo que propone el norte, que Frigate no trae | ✓ |
| **Segment-and-Track-Anything** (z-x-yang) · **Awesome-SAM2** · **Awesome-Crowd-Counting** | Tracking con SAM + listas curadas de papers | ⚠️ |
| Repos individuales de footfall: `casedone/people-counting`, `saimj7/Shopping-Analytics`, `sahillathwal/real-time-footfall-tracking`, `cnnsyhnx/CrowdInsight`, `Stereoscopic-memory180/yolov8-line-crossing-counter` | Típicamente YOLO + tracker + cruce de línea con un contador en pantalla o CSV; **sin capa temporal para EDA/ML** (a juzgar por las descripciones; no auditados) | ⚠️ |
| Comerciales: FootfallCam, Retail Sensing, Ariadne, Trakwell, Tentosoft | Conteo retail llave en mano (varios con sensores ToF, no solo visión) | ⚠️ |

## 6. Implicaciones para el lab

1. **Adoptar, no inventar, la taxonomía L2.** Usar los eventos de Google/VSS/supervision:
   `line_crossing` (con dirección), `zone_entry`/`zone_exit`, `dwell` (≥ umbral) y
   `occupancy`. Ajustar `norte.md` §3 con esos nombres cuando se implemente.
2. **SAM 3.1 como candidato de N2, con un gate de hardware primero.** Antes de integrarlo,
   medir VRAM y fps en T4 (15 GB) y en la RTX 2060: el número publicado (17,7 GB @ 5 objetos,
   H100) sugiere que no entra. Alternativas si no entra: L4/A100 para procesamiento offline,
   o quedarse con YOLO-World/Grounding DINO para el prompt en tiempo real y usar SAM 3 solo
   para refinar o etiquetar.
3. **La cascada de Alex tiene linaje académico** (NoScope/Focus/Chameleon/Rocket): citarlo al
   documentar, y seguir midiendo costo por etapa, que es lo que esa literatura valora.
4. **Evaluación de L3 con una métrica de agregación**, estilo LAVA (error del conteo agregado
   por ventana contra el GT), no solo P/R/F1 por detección.
5. **Honestidad sobre la novedad.** "Prompt → conteo por zona → series" no es nuevo: Google
   lo vendía y VSS lo implementa. Diferenciadores **candidatos**, a demostrar con datos:
   (a) open-vocab por prompt de punta a punta, con el costo medido en hardware barato;
   (b) contrato de datos L1/L2/L3 pensado para EDA/ML, no solo para alertas;
   (c) stack liviano y abierto frente a la pesadez de VSS, justo cuando se retiran los
   servicios de Google, AWS y Microsoft.
6. **Lecturas pendientes (PDF completo) antes de escribir algo publicable:** LAVA (su
   benchmark de agregación), la documentación de Behavior Analytics de VSS (esquema de
   eventos), VOCAL-UDF y el paper de SAM 3 (secciones de video y conteo en CountBench).
