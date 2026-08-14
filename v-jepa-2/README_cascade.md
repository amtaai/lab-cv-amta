# cascade de vigilancia — Semana 1: el laboratorio y el medidor

Plomería para poder responder después: **¿cuánto cuesta computacionalmente por mes
monitorear una cámara con la arquitectura de cascade, y qué tan preciso es?**

Esta semana no hay ningún modelo de detección. Hay un simulador de cámara, un
corpus versionado, el Nivel 1 (movimiento) y el medidor de costo — validado sobre
la etapa más barata posible, para que cuando lleguen las etapas caras (detección
de objetos, VLM) la instrumentación ya sea confiable.

> **El corpus arranca vacío.** La única fuente es el dataset de cámaras de
> seguridad que se está bajando aparte. No hay descarga automática de datasets ni
> material prestado de otros proyectos: se ingesta a mano con `catalog.ingest`.

## Piezas y dónde corre cada una

| Pieza | Corre en | Qué hace |
|---|---|---|
| `docker/docker-compose.yml` (`mediamtx` + `publisher`) | **Docker** | Simulador de cámara: ffmpeg loopea un clip a `rtsp://localhost:8554/cam1`. `restart: unless-stopped` da la reconexión |
| `core/cascade/catalog/` | **Docker** (`cascade`) | Esquema del corpus, índice `metadata.json`, `manifest.json` (sha256) e ingesta |
| `core/cascade/stage1_motion/` | **Docker** (`cascade`) | Nivel 1: MOG2 + guarda de flicker + warmup explícito |
| `core/cascade/cost/` | **Docker** (`cascade`) | Medidor reutilizable: `with tracker.track("etapa"): ...` → SQLite |
| `core/cascade/run_stage1.py` | **Docker** (`cascade`) | Barrido del corpus → `results/stage1_motion_{stats.json,report.md}` |

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
