# docs/ — onboarding del proyecto amta v-jepa-2

Carpeta de documentacion para que un colaborador nuevo entienda el proyecto,
sepa que se afirma como aporte (y que no), y arranque a trabajar sin re-descubrir
las decisiones ya tomadas.

> **Si solo vas a leer una cosa:** el [veredicto](04_veredicto_novedad.md). Define
> que es defendible ante un revisor y que se borra del relato. Todo lo demas se
> subordina a el.

## Que es este proyecto en una frase

Un detector de objetos sobre **cámara fija** que se auto-corrige con pseudo-labels
verificados por un VLM. **La contribucion que afirmamos NO es el lazo** (eso ya esta
publicado): es que el **error de prediccion temporal de V-JEPA 2.1** sirve como senal
de novedad open-world y de estabilidad de largo horizonte — **si y solo si** pasa un
experimento decisivo de 2-3 semanas (el GATE) contra DINOv3.

## Orden de lectura

| # | Documento | Que vas a sacar | Cuando leerlo |
|---|---|---|---|
| 0 | [`../CLAUDE.md`](../CLAUDE.md) | Topologia LOCAL/COLAB, convenciones, restriccion "sin entrenamiento local" | antes de tocar codigo |
| 1 | [`01_vision_y_plan.md`](01_vision_y_plan.md) | Idea central honesta, pregunta de investigacion, hipotesis, objetivos, GATE, delegacion de tareas e investigacion | **primero, completo** |
| 2 | [`02_arquitectura.md`](02_arquitectura.md) | 7 diagramas: topologia, percepcion, lazo, secuencia, ciclo de vida, estados, guardrails | como referencia visual del §6 del doc 1 |
| 3 | [`03_estado_del_arte.md`](03_estado_del_arte.md) | Prior-art con citas, que ya existe, el hueco defendible, cola de lectura | para no reclamar lo ya publicado |
| 4 | [`04_veredicto_novedad.md`](04_veredicto_novedad.md) | Veredicto del juez: PIVOT con gate, criterios GO/NO-GO, baselines obligatorios, venues, probabilidades | **decisorio — manda sobre el resto** |
| 5 | [`investigacion_externa/`](investigacion_externa/) | Vigilancia tecnologica: productos que absorben el Pipeline 1 (SAM 3, SAM2-OV, Grounded-SAM-2), stack metrico commodity y links | cuando dudes si algo "ya existe afuera" |
| — | [`../papers.md`](../papers.md) | URLs de referencia del usuario | consulta puntual |

## La regla de oro: primero el GATE

No se construye el sistema completo hasta correr el **Milestone 0** (detalle en
[`01_vision_y_plan.md`](01_vision_y_plan.md) §7 y [`04_veredicto_novedad.md`](04_veredicto_novedad.md) §4):
medir la senal de novedad en **C1 (DINOv3)** vs **C2 (V-JEPA 2.1)** vs **C3 (hibrido)**
a paridad de costo. Si V-JEPA no gana, se pivota o se para. Esto existe para no gastar
6 meses en algo que un experimento de 3 semanas puede descartar.

## Reparto de trabajo (resumen)

- **Lider (usuario):** direccion cientifica, criterio del gate, framing del paper, decision GO/NO-GO, custodia de `LoopThresholds` y de la honestidad del relato.
- **Colaborador A — percepcion/backbones:** la senal de novedad desde encoders congelados en las 3 condiciones (`core/perception/`).
- **Colaborador B — lazo:** VLM circuit-breaker, pseudo-labeling e instrumentacion de drift (`core/reasoning/`, `core/loop/`).

Detalle de entregables en [`01_vision_y_plan.md`](01_vision_y_plan.md) §8-9.

## Convenciones que no se negocian

- Umbrales: **siempre** desde `core/config.py::LoopThresholds`, nunca hardcodeados.
- LOCAL nunca carga V-JEPA/SAM2/VLM ni entrena: eso es Colab.
- Los 18 IDs de arxiv del proyecto fueron verificados el 2026-06-17 (todos reales; se corrigió YOLO-World a `2401.17270`). Aun así, existencia del ID ≠ claim verificado: antes de citar un resultado, abrir el PDF.
- Las frases "intuicion a nivel humano" / "mejora casi solo" **no van** en texto tecnico: son red flags para revisores.
