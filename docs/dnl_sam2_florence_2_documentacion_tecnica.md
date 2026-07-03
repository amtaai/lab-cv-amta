# Florence-2 — Documentación técnica
---

## 1. Idea central

Florence-2 convierte **cualquier tarea de visión** — clasificación, detección de objetos, segmentación, captioning, OCR, visual grounding — en el mismo problema:

> *imagen + prompt de texto → secuencia de texto de salida*

No existen cabezas (heads) separadas por tarea. Todo lo resuelve un único transformer encoder-decoder, igual que GPT resuelve traducción, resumen y preguntas con el mismo mecanismo. La tarea se selecciona con un **prompt de texto** especial:

| Prompt | Tarea activada |
|---|---|
| `<CAPTION>` | Descripción libre de la imagen |
| `<OD>` | Detección de objetos (open-vocabulary) |
| `<REFERRING_EXPRESSION_COMPREHENSION>` | Localizar objeto descrito en texto |
| `<OCR>` | Reconocimiento óptico de caracteres |
| `<OPEN_VOCABULARY_DETECTION>` | Detección de lo que el usuario describa en texto |

---

## 2. Arquitectura

```
Imagen  I ∈ R^{H×W×3}
         │
         ▼
 ┌───────────────┐
 │  Vision Encoder│  ← DaViT (Dual Attention Vision Transformer)
 │               │
 └───────┬───────┘
         │ V ∈ R^{Nv × Dv}   (tokens visuales)
         │
         ▼  proyección lineal + LayerNorm
         V' ∈ R^{Nv × D}

Prompt de texto
         │
         ▼  tokenizer + word embeddings
         T_prompt ∈ R^{Nt × D}

         X = [V', T_prompt]
         │
         ▼
 ┌───────────────────────┐
 │  Encoder-Decoder      │  ← transformer estándar
 │  Transformer          │
 └───────┬───────────────┘
         │
         ▼
  texto de salida  y = (y1, y2, ..., y|y|)
  (palabras normales + tokens de ubicación)
```

### 2.1 Vision Encoder: DaViT

DaViT (Dual Attention Vision Transformer) transforma la imagen en una secuencia plana de embeddings visuales:

$$\mathbf{I} \in \mathbb{R}^{H \times W \times 3} \;\xrightarrow{\text{DaViT}}\; \mathbf{V} \in \mathbb{R}^{N_v \times D_v}$$

Internamente, DaViT tiene **4 etapas jerárquicas** (similar a una CNN). Cada etapa empieza con una capa de patch-embedding que reduce la resolución espacial y aumenta la dimensión de canal, y luego aplica **dos tipos de atención intercalados**:

**Atención espacial por ventanas (Window Spatial Attention)**

Cada parche atiende solo a sus vecinos dentro de una ventana local de tamaño fijo $w \times w$. Dado un token $\mathbf{q}_i$ (query) y los tokens $\{\mathbf{k}_j, \mathbf{v}_j\}$ dentro de su ventana:

$$\text{Attn}(\mathbf{q}_i, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{q}_i \mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V}$$

donde $d_k$ es la dimensión de la cabeza de atención. Al operar en ventanas locales, el costo es $O(N \cdot w^2)$ en vez de $O(N^2)$ — captura detalle espacial fino eficientemente.

**Atención de canal por grupos (Channel-Group Attention)**

En vez de atender entre posiciones, "transpone" el problema: agrupa los $D$ canales en $G$ grupos de tamaño $D/G$, y dentro de cada grupo atiende entre los $N$ tokens **a lo largo de la dimensión de canal**. Esto permite que cada posición espacial capture información de contexto global (viendo el canal completo) sin el costo cuadrático en tokens. Matemáticamente, si $\mathbf{X} \in \mathbb{R}^{N \times D}$ es la entrada del bloque, se remodela como $\mathbf{X}' \in \mathbb{R}^{G \times N \times (D/G)}$ y se aplica multi-head attention sobre la dimensión $N$ para cada grupo $g$:

$$\mathbf{Y}_g = \text{MultiHeadAttn}(\mathbf{X}'_g, \mathbf{X}'_g, \mathbf{X}'_g)$$

La combinación de ambos tipos de atención en cada etapa da a DaViT tanto **sensibilidad local** (detalle espacial fino) como **comprensión global** (contexto de la imagen completa), con un costo computacional mucho menor que la atención espacial completa.

### 2.2 Fusión multimodal

Los tokens visuales $\mathbf{V} \in \mathbb{R}^{N_v \times D_v}$ tienen dimensión distinta a los tokens de texto. Se alinean mediante una proyección lineal más normalización de capa:

$$\mathbf{V}' = \text{LayerNorm}(\mathbf{W}\mathbf{V}) \in \mathbb{R}^{N_v \times D}$$

donde $\mathbf{W} \in \mathbb{R}^{D \times D_v}$. El texto del prompt se tokeniza y proyecta con un tokenizer extendido (BERT-like) a:

$$\mathbf{T}_{prompt} \in \mathbb{R}^{N_t \times D}$$

La entrada al encoder-decoder es la concatenación de ambas secuencias:

$$\mathbf{X} = [\mathbf{V}', \;\mathbf{T}_{prompt}] \in \mathbb{R}^{(N_v + N_t) \times D}$$

### 2.3 Encoder-Decoder Transformer

Sobre $\mathbf{X}$ se aplica un **encoder transformer estándar**: $L$ bloques de self-attention bidireccional + feed-forward:

$$\mathbf{H} = \text{Encoder}(\mathbf{X})$$

El **decoder** genera el texto de salida token a token, condicionado en $\mathbf{H}$: cada bloque del decoder tiene self-attention causal (atiende solo tokens ya generados) + cross-attention hacia $\mathbf{H}$:

$$\hat{y}_i = \text{Decoder}(\hat{y}_{<i}, \mathbf{H})$$

---

## 3. Representación de ubicaciones espaciales

Esta es la clave para que Florence-2 pueda hacer detección sin una cabeza especial: **convierte coordenadas en tokens de texto**.

El tokenizer se extiende con **1000 tokens de ubicación** especiales, cada uno representando un bin de una cuadrícula normalizada sobre la imagen. Una coordenada continua $x \in [0, W]$ se cuantiza así:

$$\text{loc\_token}(x) = \left\lfloor \frac{x}{W} \times 1000 \right\rfloor \in \{0, 1, \ldots, 999\}$$

Con esto, cualquier región espacial se escribe como secuencia de tokens de ubicación:

| Tipo | Formato de salida | Tarea |
|---|---|---|
| Bounding box | `<loc_x0><loc_y0><loc_x1><loc_y1>` | Detección de objetos |
| Quad box | `<loc_x0><loc_y0>...<loc_x3><loc_y3>` | OCR / texto rotado |
| Polígono | `<loc_x0><loc_y0>...<loc_xn><loc_yn>` | Segmentación por referencia |

Para el modelo, predecir una bounding box es **idéntico** a predecir la siguiente palabra de una oración: la misma operación de softmax sobre el vocabulario ampliado, en el mismo decoder. No existe ninguna diferencia arquitectónica entre generar un caption y generar coordenadas — solo cambia cuál token es el correcto en cada posición.

---

## 4. Función de pérdida

Dado que todo se reduce a generación de texto, el entrenamiento usa únicamente **cross-entropy autoregresiva**, la misma que cualquier modelo de lenguaje. Con input combinado $x$ (imagen + prompt) y secuencia objetivo $y = (y_1, \ldots, y_{|y|})$:

$$\boxed{\mathcal{L} = -\sum_{i=1}^{|y|} \log P_\theta\!\left(y_i \mid y_{<i},\, x\right)}$$

En cada paso $i$, el decoder produce una distribución sobre todo el vocabulario extendido (palabras + tokens de ubicación). El gradiente penaliza exactamente cuán lejos estuvo del token correcto $y_i$. Esta **misma ecuación** se usa para todas las tareas sin cambio alguno — la diferencia entre detectar objetos y describir una imagen es solo qué prompt se usa en $x$ y qué se pone en $y$.

---

## 5. Datos de entrenamiento: FLD-5B

Para que una sola arquitectura aprenda tareas tan distintas a la vez, se necesita un dataset con **anotaciones densas y diversas**. Microsoft construyó **FLD-5B** de forma automática:

- **126 millones de imágenes** con anotaciones de todo tipo (cajas, polígonos, captions, pares texto-región, etc.)
- **5.4 mil millones de anotaciones** en total
- Proceso de anotación en **dos fases**: (1) varios modelos especializados anotan automáticamente y se busca consenso entre ellos; (2) refinamiento iterativo donde modelos ya entrenados en el dataset filtran y mejoran las anotaciones previas, en ciclos sucesivos.

Todas las anotaciones se convierten al mismo formato texto → texto antes del entrenamiento, unificando todas las tareas bajo la misma función de pérdida.

---

## 6. Tamaños disponibles y rendimiento

| Variante | Parámetros | Uso recomendado |
|---|---|---|
| Florence-2-base | ~0.23B | Prototipos rápidos, GPU con poca VRAM |
| Florence-2-large | ~0.77B | Mejor calidad, GPU estándar (T4 en Colab) |

Pese a ser pequeño para un modelo de visión-lenguaje, Florence-2-large alcanzó resultados estado del arte en **zero-shot** en captioning (COCO), grounding visual (Flickr30k) y referring expression comprehension (RefCOCO/+/g) al momento de su publicación en CVPR 2024.

## 7. Aporte de Florence-2 al proyecto
Florence-2 actúa como el módulo de localización del pipeline: recibe el nombre del objeto a detectar en texto libre y produce la bounding box que delimita su ubicación en la imagen. Esto permite que el sistema detecte cualquier categoría de objeto sin haber sido entrenado específicamente para ella — basta con describir el objeto en texto. En el contexto del pipeline, Florence-2 es el primer paso: genera las coordenadas que SAM2 necesita para segmentar y rastrear el objeto a lo largo del video.