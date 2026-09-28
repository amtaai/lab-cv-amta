# cascade de vigilancia

Para responder: **¿cuánto cuesta computacionalmente por mes monitorear una cámara
con la arquitectura de cascade, y qué tan preciso es?**

- **Semana 1** — el laboratorio y el medidor: simulador de cámara RTSP, corpus
  versionado, Nivel 1 (movimiento) y el medidor de costo. Sin modelos: la
  instrumentación se validó sobre la etapa más barata posible para que fuera
  confiable antes de que llegaran las caras.
- **Semana 2** — Nivel 2 y conteo: se enchufa lo que **ya existía en el repo**
  —el detector de `yolo_seg/` y `yolo_world/`, ByteTrack, y `ObjectCounter` de
  `ultralytics.solutions`— a la salida del filtro de movimiento. El cascade no
  aporta modelos: aporta el gate, el cableado y la medición.
- **Evaluación** — ground truth etiquetado a mano y precision/recall/F1 barriendo
  umbrales de confianza. Ver `results/eval_report.md`.

> **El corpus arranca vacío.** La única fuente es el dataset de cámaras de
> seguridad que se está bajando aparte. No hay descarga automática de datasets ni
> material prestado de otros proyectos: se ingesta a mano con `catalog.ingest`.

## Piezas y dónde corre cada una

| Pieza | Corre en | Qué hace |
|---|---|---|
| `docker/docker-compose.yml` (`mediamtx` + `publisher`) | **Docker** | Simulador de cámara: ffmpeg loopea un clip a `rtsp://localhost:8554/cam1`. `restart: unless-stopped` da la reconexión |
| `core/cascade/catalog/` | **Docker** (`cascade`) | Esquema del corpus, índice `metadata.json`, `manifest.json` (sha256) e ingesta |
| `core/cascade/stage1_motion/` | **Docker** (`cascade`) | Nivel 1: MOG2 + guarda de flicker + warmup explícito |
| `core/cascade/stage2_person/` | **Docker** (`cascade`) | Cablea el detector del repo a la salida del filtro de movimiento |
| `core/perception/` | **Docker** (`cascade`) | Detector (YOLO11 / YOLO-World), registro de tracks y conteo |
| `core/cascade/cost/` | **Docker** (`cascade`) | Medidor reutilizable: `with tracker.track("etapa"): ...` → SQLite |
| `core/cascade/run_stage1.py` | **Docker** (`cascade`) | Barrido Nivel 1 → `results/stage1_motion_{stats.json,report.md}` |
| `core/cascade/run_stage2.py` | **Docker** (`cascade`) | Barrido cascada completa → `results/stage2_person_{stats.json,report.md}` |
| `core/cascade/stage2_person/visualizar.py` | **Docker** (`cascade`) | Dibuja cajas, IDs y confianza sobre un clip, para auditar a ojo |
| `core/perception/conteo.py` | **Docker** (`cascade`) | Conteo de personas por cruce de línea, sobre `ultralytics.solutions` |
| `core/perception/filas.py` | **Docker** (`cascade`) | Detección de fila sobre una ROI, sobre `ultralytics.solutions` |
| `core/perception/reid.py` | **Docker** (`cascade`) | Nivel 3: reengancha IDs que la oclusión partió (DINOv2 + coseno) |
| `core/cascade/eval/` | **Docker** (`cascade`) | Ground truth, P/R/F1, barrido de umbrales, análisis de errores, CSV |
| `core/api/` | **Docker** (`api`) | El pipeline servido por HTTP (FastAPI) → `http://localhost:8000/docs` |

El host tiene Python 3.14 externally-managed y sin pip: **todo el Python corre en
el contenedor**, nunca directo en la máquina.

## Flujo

1. `cd docker && docker compose build cascade`

2. **Ingestar el dataset de cámaras de seguridad.** Un clip por invocación; los
   ejes de metadata se declaran a mano porque no salen del archivo:
   ```bash
   docker compose run --rm -v /ruta/al/dataset:/in:ro cascade \
     python -m core.cascade.catalog.ingest --file /in/clip_001.mp4 \
       --source <nombre-del-dataset> \
       --camera-height ceiling --lighting bright --crowd-density sparse \
       --location-type store --license <licencia-verificada>
   ```
   Valores válidos: `--camera-height low|eye_level|ceiling`,
   `--lighting bright|dim|mixed`, `--crowd-density empty|sparse|busy`,
   `--location-type store|restaurant|entrance|checkout|other`.

   Para varios clips de golpe:
   ```bash
   for f in /ruta/al/dataset/*.mp4; do
     docker compose run --rm -v /ruta/al/dataset:/in:ro cascade \
       python -m core.cascade.catalog.ingest --file "/in/$(basename "$f")" \
         --source <dataset> --camera-height ceiling --lighting bright \
         --crowd-density sparse --location-type store --license <licencia>
   done
   ```
   Ajustar los ejes por clip: el objetivo es un corpus **deliberadamente variado**,
   no 60 clips con la misma etiqueta.

3. Levantar el simulador apuntando a un clip ya ingestado:
   ```bash
   RTSP_CLIP=clip_001.mp4 docker compose up -d mediamtx publisher
   ```

4. `docker compose run --rm cascade python -m pytest tests/cascade -v`

5. `docker compose run --rm cascade python -m core.cascade.run_stage1 --rtsp`

6. Leer `results/stage1_motion_report.md`. Al terminar: `docker compose down`.

7. **Nivel 2 + conteo.** Los pesos ya vienen en la imagen:
   ```bash
   docker compose run --rm cascade python -m core.cascade.run_stage2
   ```
   Salida en `results/stage2_person_report.md`. Con `--clips N` corre solo los
   primeros N. En GPU el barrido completo son ~20 min.

   Para usar el detector open-vocab en vez del de clases fijas:
   ```bash
   AMTA_DETECTOR=yoloworld docker compose run --rm cascade python -m core.cascade.run_stage2
   ```

8. **Mirar el resultado**, que es la única forma de auditar el seguimiento:
   ```bash
   docker compose run --rm cascade \
     python -m core.cascade.stage2_person.visualizar --clip <clip_id>
   ```
   Escribe `results/vis_<clip_id>.jpg` con las cajas, su ID y su confianza.
   Con `--video` saca el clip entero anotado en mp4. Cada `track_id` tiene un
   color fijo, así que **un cambio de color es un cambio de ID** — el error que
   ningún número agregado deja ver. Línea gruesa = track confirmado.
   En amarillo se dibuja la línea de conteo.

## Nivel 2: detección de personas y seguimiento

**El cascade no tiene detector propio.** Usa `core/perception/detector.py`, que
es el adapter importable de lo que los notebooks de `yolo_seg/` y `yolo_world/`
corren a mano en Colab. Dos backends, la misma interfaz:

| `AMTA_DETECTOR` | Modelo | Qué es |
|---|---|---|
| `yolo11` (default) | `yolo11n-seg.pt` | Clases COCO fijas. Rápido, es el que tiene sentido para un filtro por frame |
| `yoloworld` | `yolov8s-worldv2.pt` | Vocabulario abierto por texto: `prompts=["una persona"]` |

Los pesos y los umbrales son los mismos que declaran los notebooks
(`conf=0.35` para YOLO11, `conf=0.25` para YOLO-World, `iou=0.70`), para que los
resultados sean comparables con lo que ya se corrió ahí.

**El seguimiento tampoco es nuestro**: es **ByteTrack**, y no es una etapa
separable — ultralytics lo corre adentro del predictor
(`model.track(persist=True, tracker="bytetrack.yaml")`) porque necesita el frame,
no solo las cajas. El `track_id` llega junto con la caja.

Lo que sí es de este proyecto es `core/perception/tracker.py`: el **registro**.
Mantiene la historia por `track_id`, decide cuándo un track cuenta como persona
(`min_hits`) y responde `unique_count`. Está separado del detector porque el
conteo tiene reglas propias que no son las del asociador.

Un detalle que cuesta caro si se olvida: **ByteTrack guarda estado entre
llamadas**. Es lo que se quiere dentro de un clip y lo que no se quiere al pasar
al siguiente, así que el runner llama a `detector.reset()` al empezar cada clip.
Sin eso la primera persona de un clip hereda el ID de la última del anterior.

### Dónde corre y qué se mide

El Nivel 2 corre en **GPU** (RTX 2060). Eso cambia la contabilidad: el Nivel 1
gasta CPU y el Nivel 2 gasta GPU, así que **no se suman**. Son recursos distintos
con precios distintos, y un único ms/frame que los mezclara escondería cuál de
los dos es el que hay que pagar.

`gpu_time_ms` se mide con `torch.cuda.synchronize()` antes y después. Sin eso las
llamadas CUDA son asíncronas y el reloj mide el encolado, no el cómputo: el
Nivel 2 parecería costar casi cero.

La consecuencia útil es que **el ahorro de GPU es exactamente la fracción de
frames que el Nivel 1 descarta**, y eso se traduce directo en cuántas cámaras
entran por GPU.

## Conteo de personas (punto 3)

También sale de código que ya existía: `ultralytics.solutions.ObjectCounter`.
`core/perception/conteo.py` no reimplementa nada, solo cambia **de dónde saca los
tracks**.

Tal como viene, `ObjectCounter` hace `self.model.track()` en `extract_tracks()`,
o sea que detectaría por segunda vez, salteándose el filtro de movimiento y
dejando al medidor sin ver nada. Se sobreescribe ese único método para que
consuma los tracks que el cascade **ya calculó**. Todo lo demás —historia de
trayectorias, reglas de cruce, dibujo— es de ultralytics sin tocar.

### Cuándo entra, sale y se contabiliza

- **entra**: el centroide del track cruza la línea de conteo hacia la derecha
  (hacia abajo si la línea fuera horizontal). Lo decide `ObjectCounter.count_objects`.
- **sale**: el mismo cruce en sentido contrario.
- **se contabiliza**: una sola vez por `track_id` — ultralytics lleva `counted_ids`,
  así que ir y venir sobre la línea no infla el número — y solo si el registro lo
  confirmó con `min_hits` detecciones.

Ojo con dos números que miden cosas distintas: **cruzaron la línea** es tráfico
por un punto, **personas únicas** es presencia en el cuadro. No son intercambiables.

### La geometría

Va en coordenadas **normalizadas [0,1]**, nunca en píxeles: el corpus mezcla
640×480 con 1920×1080 y una línea en píxeles caería en lugares distintos en cada
una. Es el mismo error que ya se había corregido en el umbral de área del Nivel 1.

El valor por defecto **no es una elección de diseño, es dónde este corpus tiene
tráfico**. Midiendo 31 trayectorias sobre 10 clips, la gente vive entre x=0.05 y
x=0.51 y nadie llega a la mitad del cuadro, así que la línea al centro daba
**cero cruces**. Cruces por posición: x=0.25 → 3, x=0.30 → 7, **x=0.35 → 8**,
x=0.40 → 7, x=0.50 → 5. Con otro corpus hay que volver a medirlo.

> **Se quitó la detección de filas sobre una ROI.** Presuponía una zona de espera
> —el área de caja de un comercio— que los datasets de este proyecto no tienen:
> la ROI terminaba siendo un rectángulo arbitrario y su número no describía nada.
> Si aparece material de un local con caja, se reconstruye desde
> `ultralytics.solutions.QueueManager` con el mismo mixin de tracks externos que
> usa el contador.

## La API REST

El mismo pipeline servido por HTTP. No reimplementa nada: `core/api/` traduce
peticiones a llamadas del runner y respuestas del runner a JSON validado.

```bash
cd docker && docker compose up -d api      # http://localhost:8000/docs
```

`/docs` es interactivo y lo genera FastAPI desde los esquemas de
`core/api/esquemas.py`, así que nunca se desincroniza del código.

| Endpoint | Qué hace |
|---|---|
| `GET /salud` | Estado del servicio y si el modelo está disponible |
| `POST /jobs` | Inicia el procesamiento de un clip, un archivo o un stream RTSP |
| `POST /jobs/video` | Igual, pero el video viaja en la petición (multipart) |
| `GET /jobs` | Lista los procesamientos, los más nuevos primero |
| `GET /jobs/{id}` | Estado y avance en vivo de un procesamiento |
| `DELETE /jobs/{id}` | Cancela. Es la única forma de terminar un RTSP sin tope |
| `GET /jobs/{id}/resultados` | Conteo, filas, rendimiento y coste |
| `GET /jobs/{id}/metricas` | Solo rendimiento y coste, para monitoreo |
| `GET /metricas` | Agregados del servicio + el log de coste completo |

```bash
# arrancar un job y consultarlo
JOB=$(curl -s -X POST http://localhost:8000/jobs -H 'Content-Type: application/json' -d '{
  "fuente": {"tipo": "clip", "valor": "retail_iprox_caja"},
  "modelo": {"backend": "yolo11", "conf": 0.35},
  "roi": {"puntos": [[0.52,0.38],[0.98,0.38],[0.98,0.99],[0.52,0.99]]},
  "max_frames": 600
}' | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

curl -s http://localhost:8000/jobs/$JOB              # estado + progreso
curl -s http://localhost:8000/jobs/$JOB/resultados   # resultados

# un stream en vivo, con tope de 20 s
RTSP_CLIP=<clip>.mp4 docker compose up -d mediamtx publisher
curl -s -X POST http://localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"fuente":{"tipo":"rtsp","valor":"rtsp://mediamtx:8554/cam1"},"max_segundos":20}'

# subir un video
curl -s -X POST http://localhost:8000/jobs/video \
  -F 'archivo=@mi_video.mp4' -F 'peticion={"modelo":{"conf":0.4}}'
```

**Códigos de error.** `202` job aceptado y encolado · `400` fuente inutilizable ·
`404` el job, clip o archivo no existe · `409` se pidieron resultados de un job
que no terminó · `422` la petición no valida contra el esquema · `503` el
detector no está disponible.

**Un job por vez.** `max_workers=1` no es provisional: hay una sola GPU, y dos
jobs en paralelo no van al doble — se pelean la VRAM y además arruinan la
medición de ms/frame, que es uno de los entregables. Los demás esperan en cola y
eso se ve en `status`.

**Los videos subidos tienen tope y caducan.** `AMTA_API_MAX_SUBIDA_MB` (2048 por
defecto) corta la subida con `413`, y cada subida nueva borra las de más de
`AMTA_API_RETENCION_H` horas (24 por defecto). Sin lo primero una sola petición
llena el disco; sin lo segundo `uploads/` no se vacía nunca.

**El estado de los jobs vive en memoria.** Reiniciar el servicio lo borra. Lo que
NO se pierde son las mediciones de coste: se escriben en `results/costs.db` con
el `job_id` como `run_id`, así que `GET /metricas` sobrevive al reinicio aunque
`GET /jobs/{id}` no.

## Instrumentar una etapa nueva

```python
from core.cascade.cost.tracker import CostTracker

tracker = CostTracker(cfg.costs_db, cpu_usd_per_hour=cfg.cpu_usd_per_hour)
with tracker.track("object_detection", clip_id=cid) as ev:
    ...
    ev.gpu_time_ms = t_gpu      # las etapas GPU lo llenan
    ev.tokens_used = n_tokens   # las llamadas a VLM lo llenan
tracker.close()
```

Si la etapa corre **intercalada** con otra dentro del mismo loop —como el Nivel 2,
que se ejecuta salteado entre frames del Nivel 1— `track()` no sirve: los bloques
se solapan y cada context manager terminaría midiendo también el tiempo del otro.
En ese caso se acumula por fuera y se registra el total:

```python
tracker.registrar("person_detection", cpu_time_ms=ns_acumulados / 1e6,
                  n_frames=n, clip_id=cid)
```

Es también la vía para tiempos que reporta otro reloj (eventos CUDA, latencia de
una API) o para `tokens_used` de un VLM.

## Limitaciones conocidas

- **El corpus está vacío hasta que se ingeste el dataset.** `run_stage1` aborta
  con `corpus vacio` y el reporte no existe. Con menos de 60 clips de interior
  comercial el veredicto queda en `PENDIENTE` y el % de movimiento **no se
  publica**: por debajo de ese umbral el número no generaliza.
- **`cost_usd` es 0 por construcción.** `AMTA_CPU_USD_PER_HOUR` arranca en `0.0`
  a propósito. Los milisegundos de CPU sí son reales. Fijar una tarifa cloud
  verificada antes de citar cualquier costo en dólares.
- **`cpu_time_ms` usa `time.process_time_ns()`, que es del proceso entero.** Solo
  es válido en etapas secuenciales. Si alguna etapa se paraleliza hay que pasar a
  `time.thread_time_ns()` por worker, o el número queda inflado por los otros hilos.
- **`cv2.setNumThreads(1)`** se fija en `run_stage1` para que la medición sea
  reproducible. Sube el wall time y baja el paralelismo: es a propósito.
- **La guarda de flicker no separa luz de cámara.** Dispara con cualquier cambio
  global del frame. En CCTV fijo —el caso de este dataset— la cámara no se mueve,
  así que ahí sí aísla iluminación; en cámara en mano contaría movimiento de cámara.
- **El umbral `flicker_fg_ratio=0.50` no está validado contra iluminación real de
  tienda.** El test unitario prueba que el mecanismo dispara con un escalón de
  brillo sintético, no que el valor sea el correcto. Va escrito en la línea de
  protocolo del reporte para que nunca se cite el resultado sin él.
- **Los frames de warmup de MOG2 se excluyen** (`warmup_frames=30`). Sin eso MOG2
  marca casi todo como foreground al arrancar e infla el % de movimiento.
- **`-c copy` en el publisher exige que el clip sea H.264.** Si el dataset viene
  en otro códec, cambiar el comando del publisher a
  `-c:v libx264 -preset veryfast -tune zerolatency`.
- **Warnings `co located POCs unavailable` al leer el stream son esperados**: el
  decoder ve un GOP nuevo sin sus frames de referencia cada vez que el loop reinicia.
  No cortan el stream (verificado: 50 s de lectura continua cruzando el wrap).

### Limitaciones del Nivel 2

- **El conteo de personas sobre-cuenta, y bastante.** Se contaron a mano las
  personas de 4 clips (verdad: 2, 3, 1, 1); el seguidor devuelve 3, 5, 1, 1.
  Acierta los clips de una persona y **sobre-cuenta ~60 % los de varias**. La
  causa es la de siempre: el gate del Nivel 1 corta la continuidad, y la gente
  sentada —que el detector ve intermitente a conf ~0.4— recibe un ID nuevo cada
  vez que reaparece pasado `max_age_s`. **Los ms/frame son mediciones; el conteo
  de personas es una estimación con sesgo hacia arriba.**
- **Los parámetros del seguidor se ajustaron sobre esos mismos 4 clips.** n=4 no
  valida nada. `iou_min=0.15`, `max_age_s=5.0` y `min_hits=3` son un punto de
  partida razonable, no valores validados. La grilla completa está en el reporte.
- **Sin re-identificación visual.** Si una persona sale de escena y vuelve un
  minuto después, es una persona nueva. Para "cuánta gente distinta pasó hoy"
  haría falta un embedding de apariencia; para "cuánta gente hay ahora", no.
- **La asociación es voraz, no húngara.** Con las 1-5 personas de un frame de CCTV
  el óptimo global y el voraz coinciden casi siempre. Con una escena llena, no.
- **`conf_threshold=0.35` es bajo a propósito y deja pasar cajas dudosas.** Se
  probó subirlo para matar las detecciones de conf ~0.4 asumiendo que eran
  muebles; mirando los frames resultaron ser **una persona sentada de espaldas y
  cortada por el borde**. Subir el umbral —o filtrar los tracks que no se
  desplazan, que también se probó— borra gente real, que en un contador de
  personas es el peor error posible. Quedó en 0.35 a sabiendas del ruido.
- **Es CPU.** ~80 ms/frame a 640×640 con un hilo. En la 2060 sería un orden de
  magnitud menos, pero eso cambia la ecuación de costo entera (hay que amortizar
  la GPU) y el campo `gpu_time_ms` del medidor está listo para cuando se mida.

## Gotchas verificados

1. **La imagen de mediamtx es `scratch`: no tiene `sh` ni `wget`**, así que un
   `healthcheck` de Docker con `CMD sh -c` falla siempre y deja el contenedor
   `unhealthy` aunque el servidor esté perfecto. Además, desde v1.20 la API en
   `:9997` responde `401` por defecto. El readiness se chequea por TCP desde el
   código (`rtsp_sim.control.wait_until_ready`), no con un healthcheck de contenedor.
2. **En YAML, `command: >` no pliega una línea de continuación más indentada.** El
   comando del publisher quedaba partido y `sh` leía las líneas 2 y 3 como comandos
   sueltos (`/bin/sh: 2: -c: not found`). Va todo en una sola línea.
3. **`-fflags +genpts` es obligatorio con `-stream_loop -1`.** Al reiniciar el loop
   los timestamps del archivo vuelven a cero y mediamtx tira el path por PTS no
   monotónico.
4. **La db de costo acumula entre corridas.** Un reporte que sume toda la tabla
   describe una historia, no la medición de hoy. Cada `CostTracker` genera un
   `run_id` y `resumen_por_stage()` filtra por él.
5. **Los archivos que escribe el contenedor quedan `root:root`** en el host. Para
   borrar `corpus/raw/` o `results/` sin sudo, hacerlo desde el contenedor:
   `docker compose run --rm cascade sh -c 'rm -f /app/results/costs.db'`.
   O correr con `--user $(id -u):$(id -g)` desde el principio.
6. **`cv2.dnn` no puede cargar YOLOv10**: falla en el operador `TopK` de la cabeza
   end-to-end (`Can't create layer ... TopK`). De ahí que el Nivel 2 use
   onnxruntime. Un YOLOv4-tiny por darknet sí carga en `cv2.dnn`, pero mide 231
   ms/frame contra los 80 ms de YOLOv10n en onnxruntime, y con bastante menos mAP.
7. **La sesión de ONNX se crea UNA vez para todo el barrido.** Levantarla cuesta
   cientos de ms; hacerlo por clip inflaría el ms/frame del Nivel 2. Por eso
   `procesar_clip` recibe el detector inyectado.
