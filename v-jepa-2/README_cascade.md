# cascade de vigilancia — Semana 1: el laboratorio y el medidor

Plomería para poder responder después: **¿cuánto cuesta computacionalmente por mes
monitorear una cámara con la arquitectura de cascade, y qué tan preciso es?**

Esta semana no hay ningún modelo de detección. Hay un simulador de cámara, un
corpus versionado, el Nivel 1 (movimiento) y el medidor de costo — validado sobre
la etapa más barata posible, para que cuando lleguen las etapas caras (detección
de objetos, VLM) la instrumentación ya sea confiable.

## Piezas y dónde corre cada una

| Pieza | Corre en | Qué hace |
|---|---|---|
| `docker/docker-compose.yml` (`mediamtx` + `publisher`) | **Docker** | Simulador de cámara: ffmpeg loopea un clip a `rtsp://localhost:8554/cam1`. `restart: unless-stopped` da la reconexión |
| `core/cascade/catalog/` | **Docker** (`cascade`) | Esquema del corpus, índice `metadata.json`, `manifest.json` (sha256), ingesta y fetcher |
| `core/cascade/stage1_motion/` | **Docker** (`cascade`) | Nivel 1: MOG2 + guarda de flicker + warmup explícito |
| `core/cascade/cost/` | **Docker** (`cascade`) | Medidor reutilizable: `with tracker.track("etapa"): ...` → SQLite |
| `core/cascade/run_stage1.py` | **Docker** (`cascade`) | Barrido del corpus → `results/stage1_motion_{stats.json,report.md}` |

El host tiene Python 3.14 externally-managed y sin pip: **todo el Python corre en
el contenedor**, nunca directo en la máquina.

## Flujo

1. `cd docker && docker compose build cascade`
2. `docker compose up -d mediamtx publisher` — levanta el simulador RTSP.
3. Ingestar clips:
   ```bash
   docker compose run --rm -v /ruta/a/tus/clips:/in:ro cascade \
     python -m core.cascade.catalog.ingest --file /in/clip.mp4 --source <fuente> \
       --camera-height ceiling --lighting bright --crowd-density sparse \
       --location-type store --license <licencia>
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

- **El corpus no es representativo todavía.** Los 6 clips actuales son material
  interino (`location_type=other`): videos de reconstrucción 3D de `project1`, con
  cámara en mano. Por eso `run_stage1` reporta `Veredicto: PENDIENTE` en vez de un
  porcentaje. Se destraba ingiriendo ≥60 clips de interior comercial, sin tocar código.
- **Los % de movimiento de los clips interinos (53–100 %) no significan nada para
  el problema real.** Son video de cámara en mano: el frame entero se mueve, así que
  MOG2 marca casi todo como foreground. En CCTV fijo el número será mucho más bajo —
  ese es justamente el número que falta medir.
- **La guarda de flicker no separa luz de cámara.** Dispara con cualquier cambio
  global del frame. En cámara en mano cuenta movimiento de cámara, no parpadeo. En
  CCTV fijo —el caso real— la cámara no se mueve, así que ahí sí aísla iluminación.
- **`cost_usd` es 0 por construcción.** `AMTA_CPU_USD_PER_HOUR` arranca en `0.0`
  a propósito. Los milisegundos de CPU sí son reales. Fijar una tarifa cloud
  verificada antes de citar cualquier costo en dólares.
- **`cpu_time_ms` usa `time.process_time_ns()`, que es del proceso entero.** Solo
  es válido en etapas secuenciales. Si alguna etapa se paraleliza hay que pasar a
  `time.thread_time_ns()` por worker, o el número queda inflado por los otros hilos.
- **`cv2.setNumThreads(1)`** se fija en `run_stage1` para que la medición sea
  reproducible. Sube el wall time y baja el paralelismo: es a propósito.
- **El umbral `flicker_fg_ratio=0.50` no está validado contra iluminación real de
  tienda.** El test unitario prueba que el mecanismo dispara con un escalón de
  brillo sintético, no que el valor sea el correcto. Va escrito en la línea de
  protocolo del reporte para que nunca se cite el resultado sin él.
- **Los frames de warmup de MOG2 se excluyen** (`warmup_frames=30`). Sin eso MOG2
  marca casi todo como foreground al arrancar e infla el % de movimiento.
- **`-c copy` en el publisher exige que el clip sea H.264.** Un clip mpeg4 (como
  `lady-running`) no se puede publicar sin re-encodear: usar
  `-c:v libx264 -preset veryfast -tune zerolatency` si hace falta.
- **Warnings `co located POCs unavailable` al leer el stream son esperados**: el
  decoder ve un GOP nuevo sin sus frames de referencia cada vez que el loop reinicia.
  No cortan el stream (verificado: 50 s de lectura continua cruzando el wrap).

## Gotchas verificados esta semana

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
