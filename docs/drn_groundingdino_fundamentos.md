# GroundingDINO

En el notebook de SAM2-OV se usó **GroundingDINO** como detector open-vocabulary. Su tarea fue recibir un texto, por ejemplo `person, laptop, chair`, y devolver cajas aproximadas donde aparecen esos conceptos en el frame.

## Fundamento

GroundingDINO conecta lenguaje e imagen. Su objetivo es encontrar regiones visuales que correspondan a palabras o frases dadas por el usuario. A diferencia de un detector cerrado, no está limitado solo a una lista fija de clases; puede trabajar con categorías o descripciones textuales.

En nuestro flujo, GroundingDINO no generó la máscara final. Solo propuso cajas. Luego SAM2 usó esas cajas para segmentar.

## Arquitectura general

Una forma sencilla de verlo es:

```text
imagen
  ↓
extractor visual

texto
  ↓
extractor de lenguaje

imagen + texto
  ↓
mezcla visión-lenguaje
  ↓
cajas alineadas al texto
```

El modelo mira la imagen y el texto al mismo tiempo. Luego decide qué zonas visuales se relacionan mejor con las palabras del prompt.

## Funcionamiento en palabras simples

GroundingDINO funciona como un buscador visual. Tú escribes qué quieres encontrar, y el modelo marca dónde cree que está. Si le dices `person`, busca personas. Si le dices `laptop`, busca laptops. Si le dices `red backpack`, intenta encontrar una mochila roja.

Su salida principal son cajas y puntajes de confianza. Esas cajas pueden alimentar otro modelo, como SAM2, para obtener máscaras.

## Matemática esencial

GroundingDINO convierte texto e imagen en vectores numéricos. La imagen se divide en muchas regiones o características visuales, y el texto se convierte en representaciones lingüísticas.

La idea clave es la atención. La atención permite que el texto guíe qué partes de la imagen importan. En forma simplificada:

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

Interpretación simple:

```text
Q = lo que estoy buscando
K = lo que hay disponible
V = la información que puedo usar
```

Si el texto dice `laptop`, las zonas visuales que parecen laptop reciben mayor importancia.

El modelo produce cajas:

```text
bbox = (x1, y1, x2, y2)
```

y un puntaje:

```text
score = similitud(texto, región_visual)
```

Si el score supera un umbral, se acepta la detección.

## Qué aporta al proyecto

GroundingDINO permite usar lenguaje natural para decidir qué objetos buscar. Es más flexible que una lista cerrada de clases y se conecta muy bien con SAM2. En el proyecto, es la parte que entiende el prompt y transforma texto en regiones visuales.