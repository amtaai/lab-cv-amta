# Evaluacion contra ground truth — reporte

protocolo: yolo11 imgsz=640 device=0 | emparejamiento IoU>=0.5 | 31 frames, 50 personas etiquetadas (13 dificiles)

> El ground truth se etiqueto **a mano mirando los frames**, no con el modelo: si saliera del detector la evaluacion seria circular. Un solo anotador y sin segunda pasada, asi que los valores absolutos tienen el error del anotador adentro; lo que si es solido es la COMPARACION entre umbrales, que usa las mismas etiquetas para todos.

## El detector solo

Sobre los 16 frames que el Nivel 1 dejo pasar. Es lo que normalmente se reporta como precision del modelo.

### IoU >= 0.5, ignorando las cajas dificiles

| conf | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.10 | 19 | 6 | 0 | 0.760 | 1.000 | 0.864 |
| 0.20 | 19 | 2 | 0 | 0.905 | 1.000 | 0.950 |
| 0.25 | 19 | 1 | 0 | 0.950 | 1.000 | 0.974 |
| 0.30 | 19 | 1 | 0 | 0.950 | 1.000 | 0.974 |
| 0.35 | 19 | 1 | 0 | 0.950 | 1.000 | 0.974 |
| 0.40 | 19 | 1 | 0 | 0.950 | 1.000 | 0.974 |
| 0.50 | 19 | 0 | 0 | 1.000 | 1.000 | 1.000 **<-- mejor F1** |
| 0.60 | 19 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 0.70 | 19 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 0.80 | 18 | 0 | 1 | 1.000 | 0.947 | 0.973 |

## La cascada entera

Sobre los 31 frames etiquetados, incluyendo los 15 que **el Nivel 1 descarto**. Ahi el sistema no predice nada, por definicion: el detector nunca llega a correr.

### IoU >= 0.5, ignorando las cajas dificiles

| conf | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.10 | 19 | 6 | 18 | 0.760 | 0.514 | 0.613 |
| 0.20 | 19 | 2 | 18 | 0.905 | 0.514 | 0.655 |
| 0.25 | 19 | 1 | 18 | 0.950 | 0.514 | 0.667 |
| 0.30 | 19 | 1 | 18 | 0.950 | 0.514 | 0.667 |
| 0.35 | 19 | 1 | 18 | 0.950 | 0.514 | 0.667 |
| 0.40 | 19 | 1 | 18 | 0.950 | 0.514 | 0.667 |
| 0.50 | 19 | 0 | 18 | 1.000 | 0.514 | 0.679 **<-- mejor F1** |
| 0.60 | 19 | 0 | 18 | 1.000 | 0.514 | 0.679 |
| 0.70 | 19 | 0 | 18 | 1.000 | 0.514 | 0.679 |
| 0.80 | 18 | 0 | 19 | 1.000 | 0.486 | 0.655 |

El detector llega a F1 **1.000** (recall 1.000) y el sistema completo a F1 **0.679** (recall 0.514). **Toda esa diferencia es el gate**, no el modelo.

### Ojo con esta recall: la muestra esta sesgada a proposito

El muestreo pidio frames descartados por el gate para poder medirlo, asi que en la muestra pasa el **51.6 %** de los frames, contra el **63.3 %** que pasa en los 182 clips. La recall de 0.514 describe la muestra, no el corpus.

Reponderando con la tasa real —y asumiendo que la gente se reparte igual entre frames que pasan y frames que no— la recall del sistema sobre el corpus seria del orden de **0.633** (0.6331 x 1.000).

En los dos casos la lectura es la misma y es la que importa: **el techo de recall del sistema es la tasa de paso del Nivel 1**. Por bueno que sea el detector, no puede encontrar a nadie en un frame que nunca vio. El ahorro de GPU del Nivel 1 se paga en recall, y hasta ahora ese precio no estaba medido.

## Sensibilidad de la medicion

### Contando tambien las cajas dificiles (detector solo)

| conf | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.10 | 23 | 10 | 4 | 0.697 | 0.852 | 0.767 |
| 0.20 | 23 | 3 | 4 | 0.885 | 0.852 | 0.868 |
| 0.25 | 23 | 2 | 4 | 0.920 | 0.852 | 0.885 |
| 0.30 | 23 | 2 | 4 | 0.920 | 0.852 | 0.885 |
| 0.35 | 23 | 1 | 4 | 0.958 | 0.852 | 0.902 **<-- mejor F1** |
| 0.40 | 22 | 1 | 5 | 0.957 | 0.815 | 0.880 |
| 0.50 | 22 | 0 | 5 | 1.000 | 0.815 | 0.898 |
| 0.60 | 21 | 0 | 6 | 1.000 | 0.778 | 0.875 |
| 0.70 | 20 | 0 | 7 | 1.000 | 0.741 | 0.851 |
| 0.80 | 18 | 0 | 9 | 1.000 | 0.667 | 0.800 |

### Aflojando el emparejamiento a IoU >= 0.3 (detector solo)

| conf | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.10 | 19 | 5 | 0 | 0.792 | 1.000 | 0.884 |
| 0.20 | 19 | 1 | 0 | 0.950 | 1.000 | 0.974 |
| 0.25 | 19 | 1 | 0 | 0.950 | 1.000 | 0.974 |
| 0.30 | 19 | 1 | 0 | 0.950 | 1.000 | 0.974 |
| 0.35 | 19 | 1 | 0 | 0.950 | 1.000 | 0.974 |
| 0.40 | 19 | 1 | 0 | 0.950 | 1.000 | 0.974 |
| 0.50 | 19 | 0 | 0 | 1.000 | 1.000 | 1.000 **<-- mejor F1** |
| 0.60 | 19 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 0.70 | 19 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 0.80 | 18 | 0 | 1 | 1.000 | 0.947 | 0.973 |

La segunda tabla dice cuanto de lo que se ve depende de la precision con que estan dibujadas las cajas a mano. Si F1 sube mucho al aflojar el IoU, el detector encuentra a la gente pero el anotador la encuadro distinto.

## Veredicto

Con el umbral que usa el pipeline (`conf=0.35`) el detector da precision **0.950**, recall **1.000**, F1 **0.974**; el sistema completo, F1 **0.667**.

> **31 frames no alcanzan para fijar un umbral de produccion.** Son de 16 clips de la misma oficina, con las mismas personas y la misma camara. Sirven para ver la forma de la curva y para detectar un error grueso, no para elegir un valor definitivo.