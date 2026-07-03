# DINOv3 — Documentación técnica
---

## 1. Idea central

DINOv3 es radicalmente distinto a los modelos de visión-lenguaje como Florence-2: **no usa texto ni etiquetas de ningún tipo**. Es un modelo de visión puramente **auto-supervisado (self-supervised learning, SSL)** — aprende representaciones visuales de alta calidad mirando únicamente imágenes sin anotar, mediante una técnica llamada **self-distillation**: una red "estudiante" aprende a imitar a una red "maestra" que es una versión suavizada de sí misma.

No genera texto ni resuelve tareas por prompt. Su salida es un **vector de características (embedding) por cada parche de la imagen**, que se usa como representación congelada para tareas downstream (segmentación, detección, profundidad, correspondencia 3D...) entrenando solo una cabeza lineal pequeña encima — el backbone en sí no necesita ajustarse.

**La promesa del auto-aprendizaje:** sin depender de anotaciones humanas, DINOv3 puede entrenarse sobre cantidades virtualmente ilimitadas de datos y generalizarse a dominios que nunca ha visto (imágenes médicas, satelitales, biológicas...).

---

## 2. El esquema teacher-student
```
                   ┌─────────────────────────────────────────┐
                   │           Una sola imagen                │
                   └──────────┬──────────────────────────────┘
                              │
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
         recorte       recorte global        recortes locales
         global x       (para maestra)       x'₁, x'₂, ... x'₈
              │               │                   │
              ▼               ▼                   ▼
      ┌──────────────┐  ┌────────────────┐  ┌──────────────┐
      │  Estudiante  │  │    Maestra     │  │  Estudiante  │
      │  gθs         │  │    gθt         │  │  gθs         │
      └──────┬───────┘  └───────┬────────┘  └──────┬───────┘
             │                  │                   │
             └──────────────────┴───────────────────┘
                          ↕ pérdidas DINO + iBOT
```

- La **maestra** $g_{\theta_t}$ **no recibe gradientes**. Sus pesos se actualizan solo como promedio móvil exponencial (EMA) de los del estudiante:

$$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s, \quad \lambda \approx 0.9996$$

- El **estudiante** $g_{\theta_s}$ se entrena con gradiente normalmente.

- De cada imagen se generan **10 recortes**: 2 globales (alta resolución, ven ~80% de la imagen) y 8 locales (baja resolución, ven ~20%). La **maestra solo procesa los 2 globales**; el **estudiante procesa los 10**. Esto fuerza al estudiante a aprender que un recorte pequeño y parcial debe producir una representación consistente con la visión completa de la maestra — aprendiendo semántica y contexto, no solo texturas.

Ambas redes son **Vision Transformers (ViT)**:

1. La imagen se divide en parches de $P \times P$ píxeles, cada uno se aplana y se proyecta linealmente a un embedding de dimensión $D$.
2. Se añade un token especial `[CLS]` que resume la imagen completa, y **4 tokens de registro** (registers) — tokens extra sin significado espacial que actúan como memoria de trabajo y evitan artefactos de norma alta en los parches reales.
3. Se aplican $L$ bloques de multi-head self-attention, cada uno con la operación:

$$\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}} + \mathbf{B}_{\text{pos}}\right)\mathbf{V}$$

donde $\mathbf{B}_{\text{pos}}$ es el sesgo de posición introducido por **RoPE** (Rotary Position Embeddings, ver §4).

---

## 3. Las tres pérdidas de entrenamiento

La pérdida total de la primera fase combina tres objetivos:

$$\boxed{\mathcal{L}_{\text{Pre}} = \mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + 0.1 \cdot \mathcal{L}_{\text{Koleo}}}$$

### 3.1 Pérdida global — $\mathcal{L}_{\text{DINO}}$

A partir del token `[CLS]`, tanto estudiante como maestra producen una distribución de probabilidad sobre $K$ "prototipos" abstractos (no son etiquetas humanas — el modelo los descubre solo). Usando softmax con temperatura:

$$P_t = \text{softmax}\!\left(\frac{g_{\theta_t}(x)}{\tau_t}\right), \qquad P_s = \text{softmax}\!\left(\frac{g_{\theta_s}(x')}{\tau_s}\right)$$

donde $\tau_t < \tau_s$ (la maestra usa temperatura más baja, generando distribuciones más "afiladas" y por tanto más informativas como señal de supervisión). El estudiante aprende a imitar a la maestra minimizando su cross-entropy:

$$\mathcal{L}_{\text{DINO}} = -P_t(x) \cdot \log P_s(x')$$

Se calcula comparando la maestra sobre los 2 recortes globales contra el estudiante sobre todos los 10 recortes. Para evitar **colapso** (que el modelo prediga siempre el mismo prototipo), la salida de la maestra se centra antes del softmax usando el algoritmo **Sinkhorn-Knopp** (tomado de SwAV), que fuerza distribuciones uniformes a lo largo del batch — cada prototipo es usado aproximadamente igual.

El número de prototipos en DINOv3 es $K = 256{,}000$ (el doble que en DINOv2).

### 3.2 Pérdida local — $\mathcal{L}_{\text{iBOT}}$

Mismo mecanismo que DINO pero a nivel de **parche individual**, no del resumen global. Antes de pasar la imagen al estudiante, se enmascaran aleatoriamente algunos parches (como en un masked autoencoder). El estudiante debe predecir — usando los mismos prototipos que DINO — la representación que la maestra asigna a esos parches sin enmascarar:

$$\mathcal{L}_{\text{iBOT}} = -\sum_{i \in \mathcal{M}} P_t^{(i)}(x) \cdot \log P_s^{(i)}(x_{\text{masked}})$$

donde $\mathcal{M}$ es el conjunto de parches enmascarados, y $P^{(i)}$ es la distribución de probabilidad sobre prototipos para el parche $i$. Esto fuerza al modelo a aprender **características locales ricas** por parche, que son las que luego se usan en tareas densas como segmentación y estimación de profundidad.

### 3.3 Regularizador KoLeo — $\mathcal{L}_{\text{Koleo}}$

Empuja a que los embeddings dentro de un batch no colapsen en un espacio pequeño — maximiza (de forma aproximada) la distancia entre cada embedding y su vecino más cercano:

$$\mathcal{L}_{\text{Koleo}} = -\frac{1}{n}\sum_{i=1}^{n} \log\!\left(\min_{j \neq i} \lVert f_i - f_j \rVert\right)$$

donde $f_i$ son los embeddings normalizados del batch. Esto garantiza que el espacio de representación esté bien distribuido, evitando que el modelo "desperdicie" capacidad representando muchas imágenes como casi idénticas.

---

## 4. Arquitectura y escala

DINOv3 escala el backbone a **7 mil millones de parámetros** con los siguientes cambios respecto a DINOv2:

| Componente | DINOv2 (ViT-giant) | DINOv3 (ViT-7B) |
|---|---|---|
| Parámetros | 1.1B | 6.7B |
| Bloques transformer | 40 | 40 |
| Patch size | 14 px | 16 px |
| Dim. embedding | 1536 | 4096 |
| FFN type | SwiGLU | SwiGLU |
| FFN dim. oculta | 4096 | 8192 |
| Cabezas de atención | 24 | 32 |
| Embeddings de posición | Aprendidos | **RoPE axial** |
| Prototipos DINO | 128k | 256k |

**RoPE (Rotary Position Embeddings) axial.** En vez de sumar un vector de posición aprendido a cada token (que no generaliza bien a resoluciones nuevas), RoPE **rota** el vector del token en el espacio complejo según su posición $(x, y)$ en la imagen. Para dos tokens con posiciones $m$ y $n$, su producto punto después de aplicar RoPE depende solo de la diferencia de posición $m-n$, lo que hace que el modelo generalice naturalmente a imágenes de cualquier resolución o proporción. DINOv3 añade además **RoPE-box jittering**: el rango de coordenadas $[-1,1]$ se escala aleatoriamente a $[-s,s]$ con $s \in [0.5, 2]$ durante entrenamiento, para que el modelo no memorice una escala fija.

**Datos de entrenamiento.** DINOv3 parte de ~17 mil millones de imágenes públicas de Instagram, filtradas y curadas en tres partes mediante clustering jerárquico k-means en 5 niveles (usando embeddings de DINOv2 para el clustering inicial), recuperación por similitud a datasets de referencia, y datos supervisados públicos (ImageNet, Mapillary). El resultado final es un dataset curado de **1689 millones de imágenes (LVD-1689M)**. El entrenamiento usa **10% de batches homogéneos** de ImageNet-1k puro (inspirado en que batches de muy alta calidad ayudan a la estabilidad), y el resto del tiempo mezcla las tres fuentes.

**Optimización.** A diferencia de DINOv2 que usaba schedule cósmico (cosine), DINOv3 entrena con **learning rate, weight decay y momentum EMA constantes** durante 1 millón de iteraciones — esto permite continuar entrenando indefinidamente sin tener que redefinir un horizonte de optimización desde el principio. Batch size total: 4096 imágenes sobre 256 GPUs, generando 3.7M tokens por batch.

---

## 5. El problema principal que resuelve DINOv3: degradación de features densas

Al entrenar modelos grandes durante mucho tiempo, los investigadores de Meta notaron algo **contraintuitivo**:

- La **precisión de clasificación global** (medida con el token `[CLS]`) mejora indefinidamente con más entrenamiento. ✓
- Pero la **calidad de las features por parche**, necesarias para segmentación, profundidad, tracking, etc., **empieza a degradarse** después de ~200k iteraciones. ✗

Se midió visualizando la similitud coseno entre un parche de referencia y todos los demás parches de la imagen. Al principio del entrenamiento el mapa es nítido — el parche de un perro se parece a otros parches del perro y no a los del fondo. Después de 600k-1M iteraciones, el mapa se vuelve ruidoso: parches lejanos e irrelevantes muestran alta similitud con el parche de referencia.

**Causa identificada:** el token `[CLS]` (resumen global) y los tokens de parche individuales se van pareciendo cada vez más a medida que avanza el entrenamiento. Los parches "pierden su identidad local" porque el objetivo global (DINO sobre `[CLS]`) termina dominando y arrastrando todos los features hacia un espacio más global.

---

## 6. La solución: Gram Anchoring

La idea central: en vez de regular directamente los vectores de cada parche (lo que interferiría con el aprendizaje global), se regula la **estructura de relaciones entre parches** — su matriz de Gram.

**Formulación matemática.** Para una imagen con $P$ parches, sea $\mathbf{X}_S \in \mathbb{R}^{P \times d}$ la matriz de features $L_2$-normalizadas del **estudiante**, y $\mathbf{X}_G \in \mathbb{R}^{P \times d}$ la del **"Gram teacher"** — una copia del modelo tomada en una iteración temprana del entrenamiento (ej. iteración 200k), cuando sus features densas todavía eran de alta calidad.

La **matriz de Gram** de cada modelo es el conjunto de todos los productos punto entre parches:

$$\mathbf{G}_S = \mathbf{X}_S \mathbf{X}_S^{\top} \in \mathbb{R}^{P \times P}, \qquad \mathbf{G}_G = \mathbf{X}_G \mathbf{X}_G^{\top} \in \mathbb{R}^{P \times P}$$

Cada entrada $(\mathbf{G})_{ij}$ es la **similitud coseno entre el parche $i$ y el parche $j$** (ya que los features están normalizados en $L_2$). La pérdida de Gram anchoring empuja a que la estructura de similitudes del estudiante se parezca a la del Gram teacher:

$$\boxed{\mathcal{L}_{\text{Gram}} \;\propto\; \big\lVert \mathbf{G}_S - \mathbf{G}_G \big\rVert_F^2 = \sum_{i,j}\!\left((\mathbf{X}_S \mathbf{X}_S^{\top})_{ij} - (\mathbf{X}_G \mathbf{X}_G^{\top})_{ij}\right)^2}$$

La pérdida de la segunda fase de entrenamiento (fase de "refinamiento") añade este término:

$$\mathcal{L}_{\text{Ref}} = \mathcal{L}_{\text{Pre}} + \mathcal{L}_{\text{Gram}}$$

**¿Por qué funciona?** Operar sobre la matriz de Gram y no sobre los features directamente tiene una propiedad crucial: el estudiante sigue siendo libre de mover sus vectores de features a donde el aprendizaje global lo requiera, **siempre y cuando preserve la misma estructura de qué-parche-se-parece-a-qué-parche** que tenía el modelo cuando sus features densas eran buenas. El Gram teacher no le dice al estudiante "tus features deben ser exactamente estos vectores" — le dice "los parches del perro deben seguir siendo más parecidos entre sí que a los parches del fondo". Esto desacopla perfectamente el objetivo global del objetivo denso.

**Extensión a alta resolución — $\mathcal{L}_{\text{HRef}}$.** El Gram teacher también procesa las imágenes a mayor resolución, donde los mapas de similitud son más nítidos. Esas features de alta resolución se submuestrean para igualar la resolución del estudiante, y se usa la matriz de Gram resultante como objetivo adicional. Esto destila la consistencia espacial de alta resolución hacia el estudiante que entrena a resolución estándar.

---

## 7. Post-entrenamiento: distilación y familia de modelos

El ViT-7B es potente pero caro. DINOv3 introduce una **destilación multi-estudiante eficiente**: el maestro de 7B se mantiene congelado y se entrenan **varios modelos más pequeños en paralelo**, todos en el mismo paso de optimización. Esto es más eficiente que destilaciones secuenciales porque la inferencia del maestro se comparte entre todos los estudiantes.

Los modelos resultantes de la familia DINOv3:

| Modelo | Tipo | Parámetros | Nota |
|---|---|---|---|
| ViT-7B | ViT (maestro) | 6.7B | Modelo base, costoso |
| ViT-L | ViT | ~300M | El más usado en práctica; rendimiento cercano al 7B |
| ViT-B | ViT | ~86M | Balance velocidad/calidad |
| ViT-S | ViT | ~22M | Edge devices, tiempo real |
| ConvNeXt-L/B/S | ConvNet | varios | Alternativa CNN para deployment sin transformers |

Además se aplican dos pasos adicionales:

- **High-resolution fine-tuning**: 10k iteraciones adicionales con resoluciones mixtas globales (512, 768) y locales (112–336), con Gram anchoring del maestro 7B — hace el modelo agnóstico a la resolución de inferencia.
- **dino.txt**: alineación post-hoc con texto usando destilación desde un modelo CLIP, añadiendo capacidad zero-shot por texto sin haber entrenado con texto desde el principio.

---

## 8. Resultados destacados

Con el backbone **congelado** (sin ningún fine-tuning), DINOv3 superó en 2025 a todos los modelos previos en tareas densas:

| Tarea | Métrica | DINOv2 | DINOv3 | Ganancia |
|---|---|---|---|---|
| Segmentación semántica ADE20k (linear probe) | mIoU | 49.5 | **55.9** | +6.4 pts |
| Profundidad monocular NYUv2 (linear probe) | RMSE | 0.372 | **0.309** | −17% error |
| Detección COCO (con cabeza ligera) | mAP | — | **66.1** | estado del arte |
| Segmentación semántica ADE20k (con cabeza) | mIoU | — | **63.0** | supera modelos afinados |

La mejora en ADE20k de +6 puntos sobre DINOv2 es atribuida directamente a Gram anchoring, validado en ablaciones donde quitar ese componente devuelve el rendimiento al nivel de DINOv2.


## 9. Aporte de DINOv3 al proyecto

DINOv3 funciona como backbone de extracción de características visuales: procesa la imagen y produce un vector de representación denso por cada parche, capturando tanto información local (textura, bordes, forma) como contexto global de la escena. Estas representaciones pueden usarse directamente como entrada para un detector o clasificador ligero entrenado encima, sin necesidad de ajustar el modelo base. Su principal ventaja para detección de objetos es que al haberse entrenado de forma auto-supervisada sobre miles de millones de imágenes, generaliza bien a objetos y escenas que no aparecen en datasets de entrenamiento convencionales.


