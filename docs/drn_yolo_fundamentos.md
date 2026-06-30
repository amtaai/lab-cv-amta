# YOLO

En los notebooks se usó la familia **YOLO** en dos formas: **YOLO11-seg** para segmentación por instancias y **YOLO-World** para detección abierta por texto. Aunque tienen diferencias, ambos comparten una idea central: procesar la imagen de manera eficiente para localizar objetos rápidamente.

## Fundamento

YOLO significa **You Only Look Once**. La idea es mirar la imagen una sola vez y producir directamente las predicciones: dónde está el objeto, qué clase parece ser y, en el caso de segmentación, qué píxeles pertenecen a cada instancia. Esto lo hace adecuado para video porque evita procesos muy lentos frame por frame.

En **YOLO11-seg**, el modelo trabaja con clases conocidas y devuelve cajas, clases y máscaras. En **YOLO-World**, el modelo recibe una lista de textos como `person`, `laptop` o `bottle`, y busca objetos que coincidan con esos conceptos.

## Arquitectura general

Una forma sencilla de verlo es:

```text
imagen
  ↓
extractor visual
  ↓
combinador de información a varias escalas
  ↓
cabeza de predicción
  ↓
cajas, clases y/o máscaras
```

El extractor visual aprende patrones simples y complejos: bordes, formas, texturas, partes de objetos. Luego se combinan señales de diferentes tamaños para detectar objetos pequeños, medianos y grandes. Finalmente, la cabeza de predicción produce la salida.

En YOLO11-seg, además de la caja, se genera una máscara para cada objeto. En YOLO-World, se agrega una comparación entre regiones visuales y textos definidos por el usuario.

## Funcionamiento en palabras simples

YOLO11-seg mira un frame y dice: “aquí hay una persona, esta es su caja, y esta es su silueta”. YOLO-World dice: “de esta lista de cosas que me diste, creo que esta región se parece a `person`, esta a `laptop`, y esta a `cup`”. En ambos casos, el modelo convierte una imagen en resultados útiles para video.

## Matemática esencial

Una imagen se representa como una matriz de números:

```text
alto × ancho × canales
```

El modelo aplica filtros aprendidos, llamados convoluciones, para producir mapas de características. Esos mapas guardan información visual útil.

Para detección, se predicen cajas:

```text
bbox = (x, y, w, h)
```

o:

```text
bbox = (x1, y1, x2, y2)
```

También se predice una confianza:

```text
confidence ∈ [0, 1]
```

En segmentación, se predice una máscara:

```text
mask[y, x] = probabilidad de que el píxel pertenezca al objeto
```

Luego se convierte en máscara binaria:

```text
mask_binaria = mask > threshold
```

Durante entrenamiento, el modelo minimiza errores de localización, clasificación y segmentación:

```text
loss_total = loss_caja + loss_clase + loss_máscara
```

En YOLO-World aparece otra idea: texto e imagen se convierten en vectores. El modelo compara una región visual contra un texto usando una medida de similitud:

```text
similitud = vector_imagen · vector_texto
```

Si la región visual se parece al texto, la puntuación sube.

## Qué aporta al proyecto

YOLO es la opción más práctica para un MVP porque es rápido, fácil de ejecutar en Colab y suficiente para obtener detecciones, máscaras o cajas sobre video. Para análisis de movimiento, su salida puede alimentar tracking, series de tiempo y clusterización.