# Analisis de errores — reporte

protocolo: yolo11 conf=0.35 IoU match=0.5 | ground truth: 31 frames, 50 personas | tracks: 3 clips

## 1. Falsos positivos

**1** cajas predichas sin persona real detras, sobre 16 frames analizados.

| frame | confianza | caja (normalizada) |
|---|---|---|
| `normal_normal-87__f0319` | 0.467 | (0.371, 0.249, 0.486, 0.604) |

Son pocos y de confianza baja: el detector casi no inventa gente. Subir el umbral los elimina sin costo de recall.

## 2. Personas no detectadas

Hay que separar las dos causas, porque se arreglan en lugares distintos:

- **el detector no la vio**: 0 casos
- **el Nivel 1 descarto el frame**: 18 casos — el detector nunca llego a correr

La segunda es la dominante y **no se arregla tocando el detector**. Es el precio del gate: el techo de recall del sistema es su tasa de paso. Frames afectados:

- `normal_normal-34__f0061`
- `normal_normal-62__f0127`
- `normal_normal-63__f0192`
- `normal_normal-6__f0144`
- `normal_normal-80__f0075`
- `normal_normal-87__f0272`
- `normal_normal-9__f0088`
- `shoplifting_shoplifting-31__f0169`

## 3. Doble conteo

| clip | IDs sin ReID | IDs con ReID | de mas | reenganches |
|---|---|---|---|---|
| retail_iprox_caja | 28 | 9 | 19 | 24 |
| retail_iprox_tienda | 134 | 26 | 108 | 131 |
| retail_usa_tienda | 45 | 12 | 33 | 42 |

La columna *de mas* es doble conteo puro: la misma persona abierta varias veces. El Nivel 3 la recupera reconociendo la apariencia.

## 4. Perdida de seguimiento

Un track muere y otro nuevo aparece en el mismo lugar poco despues: la persona siguio ahi pero cambio de identidad. Se cuenta con una ventana de 45 frames e IoU >= 0.25.

| clip | relevos sin ReID | relevos con ReID |
|---|---|---|
| retail_iprox_caja | 7 | 0 |
| retail_iprox_tienda | 40 | 0 |
| retail_usa_tienda | 13 | 0 |

## 5. Oclusiones

Se cuenta un track como ocluido si alguna vez se piso con otro por encima de IoU 0.3.

| clip | tracks con oclusion | de un total de |
|---|---|---|
| retail_iprox_caja | 8 | 9 |
| retail_iprox_tienda | 22 | 26 |
| retail_usa_tienda | 9 | 12 |

Es la causa de fondo de las categorias 3 y 4: cuando dos personas se pisan, el seguidor por posicion no tiene con que distinguirlas al separarse. Por eso el Nivel 3 usa apariencia y no posicion.

## Veredicto

El error dominante **no es del detector**: son las 18 personas que el Nivel 1 descarto contra 0 que el detector no vio, y 1 falsos positivos. La segunda fuente es el doble conteo por oclusion, que el Nivel 3 reduce pero no elimina.

> El ground truth son 31 frames de una misma oficina y el analisis de tracks, 3 clips de tienda. Alcanza para ordenar las causas por importancia, no para poner un numero de produccion en cada una.