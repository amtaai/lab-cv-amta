# docs/investigacion_externa/ — competidores y componentes externos

Carpeta de **vigilancia tecnológica**: productos y modelos publicados que tocan
nuestro espacio de problema. Sirve para dos cosas:

1. Saber **qué ya está resuelto fuera** (y por tanto NO es nuestro aporte) — para
   no reescribir commodities ni reclamar lo ya publicado.
2. Tener los **links a mano** con su rol mapeado contra nuestros dos pipelines.

> Regla de honestidad (igual que en [`../README.md`](../README.md)): existencia
> de un ID de arxiv ≠ claim verificado. Antes de citar un resultado de cualquiera
> de estos papers, **abrir el PDF**. Lo verificado por fetch en esta tanda está
> marcado; lo demás es "a leer".

## Mapa: qué absorbe cada quién

Nuestro sistema tiene dos pipelines (ver [`competidores_y_componentes.md`](competidores_y_componentes.md)
para el detalle):

- **Pipeline 1 — espacial "detectá X":** detector open-vocab → SAM2 máscaras →
  tracker IDs. Continuo, tiempo real. **Es commodity.**
- **Pipeline 2 — temporal "caracterizá el comportamiento":** señal de novedad/
  sorpresa de V-JEPA en tiempo real → VLM (Qwen) interpreta on-demand. **Aquí
  vive el aporte**, junto con el lazo de auto-corrección y sus guardrails.

| Producto / modelo | Cubre Pipeline 1 | Cubre Pipeline 2 | Veredicto |
|---|---|---|---|
| **SAM 3 / 3.1** | Sí (detect+segment+track desde concepto) | No | Front-end ideal de P1 |
| **SAM2-OV** | Sí (open-vocab MOT) | No | Absorbe P1, nada de razonamiento |
| **Grounded-SAM-2** | Sí (Grounding DINO + SAM2) | No | Absorbe P1 |
| **SAM2MOT / SLAck / OpenWorldSAM** | Sí (variantes de tracking OV) | No | Absorben P1 |
| **Ultralytics YOLO + BoT-SORT/ByteTrack** | Sí (detect+track+conteo métrico) | No | Stack métrico commodity |
| **V-JEPA 2** | No (no da cajas) | Parcial (encoder temporal) | Pieza nuestra de P2 |

**Conclusión que ordena el proyecto:** todo el ecosistema SAM3/OV te resuelve el
Pipeline 1 gratis. El aporte defendible NO puede ser detectar+segmentar+seguir
"X" — eso es commodity. El aporte vive en (a) la **señal temporal de novedad**,
(b) el **razonamiento semántico de comportamiento (VQA on-demand)**, y (c) el
**lazo cerrado de auto-corrección con guardrails de drift**.

## Archivos

- [`competidores_y_componentes.md`](competidores_y_componentes.md) — síntesis de
  los hallazgos + todos los links agrupados por rol.

## Relación con el resto de docs

- El veredicto de novedad ([`../04_veredicto_novedad.md`](../04_veredicto_novedad.md))
  manda: si el GATE dice NO-GO sobre la señal temporal de V-JEPA, gran parte de
  lo que aquí justifica el aporte se cae.
- El estado del arte ([`../03_estado_del_arte.md`](../03_estado_del_arte.md))
  tiene el prior-art con citas verificadas; esta carpeta es su complemento de
  "productos que compiten/absorben", no lo reemplaza.
