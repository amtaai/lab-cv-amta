# Cascade Nivel 1 (motion detection) — reporte

protocolo: MOG2 history=500 varThreshold=16.0 min_area_px=500 warmup=30 flicker_fg_ratio=0.5 | cv_threads=1

## Por clip

| clip_id | tipo | frames | % movimiento | % descartado | flicker | cpu_ms |
|---|---|---|---|---|---|---|
| Nuevodataset | other | 964 | 77.73 | 22.27 | 108 | 49010.8 |
| collitions | other | 715 | 57.52 | 42.48 | 275 | 42855.4 |
| lady-running | other | 65 | 100.00 | 0.00 | 0 | 1094.8 |
| primeraPersona | other | 1023 | 77.74 | 22.26 | 201 | 57521.0 |
| responsive | other | 332 | 53.31 | 46.69 | 14 | 9904.5 |
| scannet_scannetpp_sample | other | 338 | 100.00 | 0.00 | 0 | 14868.3 |

## Costo (Nivel 1)

- **motion_detection**: 7 eventos | cpu medio 26903.73 ms/evento | cpu total 188326.1 ms | costo USD 0.00000000
- **por frame**: 50.99 ms cpu/frame (3437 frames de archivo, sin contar el stream RTSP)

> `AMTA_CPU_USD_PER_HOUR` esta en 0.0, asi que **cost_usd es 0 por construccion**. Los tiempos de CPU si son reales. Fijar una tarifa cloud verificada antes de citar cualquier costo en dolares.

> La guarda de flicker dispara con cualquier cambio global del frame, y no puede separar un cambio de luz de un movimiento de camara. En clips de camara en mano el contador de flicker mide lo segundo. En CCTV fijo —el caso real— la camara no se mueve, asi que ahi si aisla iluminacion.

## Veredicto

**PENDIENTE — corpus no representativo.** Hay 0 clips de interior comercial sobre un objetivo de 60. El % de frames con movimiento relevante NO se publica: los clips disponibles son material interino (`location_type=other`) y su porcentaje no generaliza a una tienda o un restaurante. La instrumentacion de costo SI esta validada.