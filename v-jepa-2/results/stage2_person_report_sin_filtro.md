# Cascade Nivel 2 (deteccion de personas + seguimiento) — reporte

protocolo: yolo11 pesos=(default) imgsz=640 conf=0.35 iou=0.7 device=0 prompts=['person'] | seguimiento bytetrack.yaml min_hits=3 max_age=30

conteo: linea=[(0.35, 0.0), (0.35, 1.0)] (normalizada)

Nivel 3 (ReID): facebook/dinov2-small umbral=0.5 ventana=30.0s mascara=False

gate Nivel 1: MOG2 min_area_frac=0.0016 warmup=30 flicker_fg_ratio=0.5 | cv_threads=1

> El detector y el tracker NO son de este modulo: son `core/perception/`, los mismos que corren los notebooks de `yolo_seg/` y `yolo_world/`. El cascade los cablea al filtro de movimiento y les mide el costo.

## Por clip

| clip_id | frames | % al Nivel 2 | personas | entra | sale | conf media | ms gpu/frame |
|---|---|---|---|---|---|---|---|
| retail_iprox_caja | 2576 | 100.00 | 9 | 2 | 1 | 0.773 | 10.9 |
| retail_iprox_tienda | 1452 | 100.00 | 26 | 9 | 7 | 0.623 | 11.8 |
| retail_usa_tienda | 1810 | 100.00 | 13 | 3 | 4 | 0.668 | 11.4 |

## Deteccion

- frames analizados por el Nivel 2: **5838** de 5838 utiles (100.00 %)
- frames con al menos una persona: **5838** (100.00 % de los analizados)
- cajas totales: **24557** | confianza media **0.678**

## Seguimiento

- personas unicas (tracks confirmados): **48** sumando los 3 clips
  - El tracker se reinicia en cada clip, que es lo correcto: son grabaciones distintas. Asi que este total son personas-por-clip, no humanos distintos; el mismo actor reaparece en muchos clips.
- cajas por persona: **511.6** — sin seguimiento esas 24557 cajas se contarian como 24557 personas
- la asociacion la hace **bytetrack.yaml** adentro de ultralytics; un track cuenta como persona a partir de 3 detecciones
- el Nivel 3 recupero **203 identidades** que la oclusion habia partido. Sin el, cada vez que alguien se tapa detras de una gondola y reaparece cuenta como una persona nueva.

> **Cuanto vale este numero.** El conteo no esta validado contra etiquetas: no hay ground truth de cuantas personas hay en cada clip del corpus. Los ms/frame de este reporte son mediciones; el conteo de personas es una salida del sistema sin verificar. Validarlo pide etiquetar a mano una muestra de clips.

## Conteo de personas

Las tres reglas, para que el numero se pueda auditar:

- **entra**: el centroide del track cruza la linea de conteo hacia la derecha (o hacia abajo si la linea fuera horizontal)
- **sale**: el mismo cruce en sentido contrario
- **se contabiliza**: una sola vez por `track_id`, y solo si el registro lo confirmo con 3 detecciones. Ir y venir sobre la linea no infla el numero

- entradas: **14** | salidas: **12**
- personas distintas que cruzaron la linea: **26**
- personas distintas vistas en el cuadro (crucen o no): **48**

Los dos ultimos numeros miden cosas distintas y conviene no confundirlos: el primero es trafico por un punto y el segundo es presencia. La diferencia entre ambos es gente que se movio en el cuadro sin llegar a cruzar la linea.

## Costo por etapa

- **conteo**: 3 eventos | cpu 6650.2 ms | gpu 0.0 ms | costo USD 0.00000000
- **motion_detection**: 3 eventos | cpu 127024.1 ms | gpu 0.0 ms | costo USD 0.00000000
- **person_detection**: 3 eventos | cpu 0.0 ms | gpu 65960.0 ms | costo USD 0.00000000
- **reid**: 3 eventos | cpu 0.0 ms | gpu 8571.8 ms | costo USD 0.00000000
- **tracking**: 3 eventos | cpu 164.1 ms | gpu 0.0 ms | costo USD 0.00000000

| etapa | recurso | corre sobre | ms totales | ms/frame |
|---|---|---|---|---|
| Nivel 1 · MOG2 | CPU | los 5838 frames | 127024 | 21.76 |
| Nivel 2 · yolo11 | GPU | los 5838 con movimiento | 65960 | 11.30 |
| registro de tracks | CPU | los 5838 con movimiento | 164 | 0.028 |
| conteo por linea | CPU | los 5838 con movimiento | 6650 | 1.139 |
| Nivel 3 · ReID | GPU | solo cuando aparece un ID nuevo | 8572 | 1.468 |

> `AMTA_GPU_USD_PER_HOUR` esta en 0.0, asi que **el costo en dolares del Nivel 2 es 0 por construccion**. Los milisegundos de GPU si son reales. Fijar una tarifa verificada antes de citar dolares.

## Economia de la cascada

- C1 (Nivel 1, CPU) = **21.76 ms/frame**, sobre el 100 % de los frames
- C2 (Nivel 2, GPU) = **11.30 ms/frame**, solo sobre los que pasan
- tasa de paso medida p = **1.0000**

- GPU si el detector corriera sobre todo: 66 s
- GPU en cascada: 66 s
- **ahorro de GPU: 0.0 %**

Con las etapas en recursos distintos, la cascada ya no es un intercambio de ms contra ms: el Nivel 1 gasta CPU, que sobra, para no gastar GPU, que es el recurso caro y el que limita cuantas camaras entran por maquina. El ahorro de GPU es directamente proporcional a los frames que el Nivel 1 descarta.

## Cuantas camaras entran por GPU

- escena vacia : n=0 clips, p_v = 0.0000
- escena activa: n=3 clips, p_a = 1.0000

| actividad del dia | tasa de paso | ms gpu/frame | ahorro de GPU | camaras por GPU a 25 fps |
|---|---|---|---|---|
| 5 % | 0.0500 | 0.56 | 95.0 % | 70.8 |
| 10 % | 0.1000 | 1.13 | 90.0 % | 35.4 |
| 15 % | 0.1500 | 1.69 | 85.0 % | 23.6 |
| 25 % | 0.2500 | 2.82 | 75.0 % | 14.2 |
| 50 % | 0.5000 | 5.65 | 50.0 % | 7.1 |

Sin cascada entran **3.5 camaras** por GPU, sin importar si hay alguien o no. Esa es la fila contra la que hay que comparar.

## Veredicto

El Nivel 2 corre sobre el **100.00 %** de los frames (el resto lo descarta el Nivel 1) y encuentra **48 personas unicas** en 24557 cajas, con confianza media 0.678. La cascada ahorra **0.0 %** de GPU contra correr el detector siempre. Cruzaron la linea de conteo **26** personas (14 entradas, 12 salidas).

> **ALCANCE.** 0 de los 3 clips estan indexados como `location_type=other`: NO son interiores comerciales. Las cifras describen el corpus disponible, no un comercio. Los ms/frame de cada etapa si son transferibles —dependen del hardware y de la resolucion, no del contenido—; lo que no transfiere es la tasa de paso p, y para eso esta la tabla por ciclo de actividad.