# SAM / SAM2

En el notebook de SAM2-OV se usó **SAM2.1** como modelo de segmentación. Su rol no fue detectar objetos por texto directamente, sino recibir una pista visual, como una caja, y convertirla en una máscara fina.

## Fundamento

SAM significa **Segment Anything Model**. La idea principal es segmentar casi cualquier objeto si se le da una pista: una caja, un punto o una máscara previa. SAM2 extiende esta idea a video, permitiendo segmentar objetos en imágenes y mantener información temporal cuando se usa su modo de video.

En nuestro caso, SAM2 recibió cajas producidas por GroundingDINO y las transformó en máscaras más precisas.

## Arquitectura general

Una forma simple de verlo es:

```text
imagen
  ↓
codificador de imagen

pista visual: caja, punto o máscara
  ↓
codificador de prompt

imagen + prompt
  ↓
decodificador de máscara
  ↓
máscara final
```

El codificador de imagen resume visualmente la escena. El codificador de prompt convierte la pista visual en información que el modelo puede usar. Luego el decodificador combina ambas cosas y produce la máscara.

En SAM2 para video, también existe una memoria temporal. Esa memoria ayuda a recordar el objeto entre frames, aunque en nuestro notebook usamos una versión frame por frame más simple.

## Funcionamiento en palabras simples

SAM2 funciona como un recortador inteligente. Si se le da una caja alrededor de una persona, intenta pintar exactamente la silueta de esa persona. No necesita saber si es persona, silla o botella. Solo necesita una señal que diga: “segmenta esto”.

Por eso SAM2 combina muy bien con detectores abiertos por texto: otro modelo encuentra el objeto y SAM2 lo recorta con más detalle.

## Matemática esencial

La imagen se transforma en una representación numérica:

```text
imagen → embedding_visual
```

La pista visual también se transforma en números:

```text
caja/punto/máscara → embedding_prompt
```

Luego el modelo calcula:

```text
máscara = f(embedding_visual, embedding_prompt)
```

La salida es una matriz de probabilidades:

```text
mask ∈ R^(H×W)
```

Cada posición indica qué tan probable es que ese píxel pertenezca al objeto:

```text
mask[y, x] = 0.93  → muy probablemente objeto
mask[y, x] = 0.02  → muy probablemente fondo
```

Después se aplica un umbral:

```text
mask_binaria = mask > threshold
```

Para aprender buenas máscaras, se usan pérdidas de segmentación. Una idea común es medir cuánto se parece la máscara predicha a la real. Una métrica intuitiva es Dice:

```text
Dice = 2 × intersección / (área_predicha + área_real)
```

Si las máscaras coinciden bien, la intersección es grande y el valor mejora.

## Qué aporta al proyecto

SAM2 aporta máscaras más finas que muchos segmentadores rápidos. Es útil si se necesita precisión en bordes, siluetas o separación visual clara. Para movimiento puro, no siempre es necesario; para análisis visual detallado, sí aporta bastante.