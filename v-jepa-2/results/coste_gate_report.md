# Coste con y sin el filtro de movimiento — reporte

Las dos corridas son sobre los mismos clips y con el mismo detector. La unica diferencia es si el Nivel 1 decide que frames llegan al detector o si le llegan todos.

| | con cascada | sin cascada | diferencia |
|---|---|---|---|
| frames del video | 5,838 | 5,838 | +0.0 % |
| frames que vio el detector | 4,965 | 5,838 | -15.0 % |
| detecciones producidas | 21,659 | 24,557 | -11.8 % |
| CPU (ms) | 128,705 | 133,838 | -3.8 % |
| GPU (ms) | 60,977 | 74,532 | -18.2 % |
| reloj de pared (s) | 200.0 | 220.0 | -9.1 % |
| FPS del pipeline | 29.2 | 26.5 | +10.0 % |

**El filtro ahorra 18.2 % de GPU** y el pipeline corre 1.10x mas rapido (29.2 contra 26.5 FPS).

### El intercambio, en milisegundos por frame

- el gate **cuesta 21.1 ms de CPU** en cada frame, porque MOG2 corre sobre todos
- y **ahorra 2.3 ms de GPU** por frame

No es un buen trato en tiempo total —se gasta mas CPU de la que se ahorra en GPU— y aun asi conviene, porque **los dos recursos no valen lo mismo**: la GPU es la que limita cuantas camaras entran por maquina y la CPU suele sobrar. El intercambio es deliberado.

A 720p MOG2 sale caro: 21.1 ms/frame contra los 8,9 ms medidos sobre el corpus completo, que es casi todo 480p. A mayor resolucion el gate se encarece y habria que medir de nuevo si sigue conviniendo.

### Por que la CPU total tambien baja

La tabla muestra menos CPU con cascada, lo que confunde: MOG2 corre sobre todos los frames en las DOS corridas. Lo que de verdad se ahorra son 1,513 ms de registro de tracks y conteo, que tampoco corren sobre los frames descartados. El resto de la diferencia es ruido de medicion entre corridas (3,620 ms en MOG2, 2.9 %).

> La CPU se mide con `time.thread_time_ns()`. Con `process_time_ns()`, que es lo que usaba la Semana 1, el mismo trabajo de MOG2 daba 133.268 y 153.621 ms segun cuanto trabajara el detector en paralelo: process_time cuenta los hilos que levanta torch y los sumaba al Nivel 1.

## Coste en dolares

> Las dos tarifas estan en 0.0, que es el default a proposito, asi que **el coste en dolares es 0 por construccion**. Los milisegundos si son reales. Con `AMTA_CPU_USD_PER_HOUR` y `AMTA_GPU_USD_PER_HOUR` fijadas a una tarifa verificada, esta tabla se llena sola:

| | con cascada | sin cascada |
|---|---|---|
| USD de CPU | 0.00000000 | 0.00000000 |
| USD de GPU | 0.00000000 | 0.00000000 |

## Que se pierde

- personas contadas: **49** con cascada contra **48** sin ella
- detecciones: 21,659 contra 24,557

El ahorro no es gratis: los frames que el Nivel 1 descarta pueden tener gente. Cuanto cuesta eso en recall esta medido aparte, en `eval_report.md` y en `errores_report.md`.