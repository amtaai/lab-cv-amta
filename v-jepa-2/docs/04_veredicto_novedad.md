# 04 - Veredicto de novedad y publicabilidad

> Juez imparcial. Veredicto sobre si vale la pena ejecutar y publicar este proyecto,
> con criterios cuantitativos y un gate temprano de decisión. Sin adornos.
> Fecha: 2026-06-17. Base de evidencia: 4 agentes de investigación (jun 2026).

---

## 1. Veredicto en una línea

**PIVOT** — con un GATE de 2-3 semanas que decide si el pivote es viable o es NO-GO.

No es GO porque el proyecto *tal como está planteado* (lazo completo world-model +
SAM2 + NL + guardrails + destilación + mall) es, ante un revisor, "integración de
APIs ya publicadas". No es NO-GO porque hay UN ángulo defendible que nadie cerró:
**detección de novedad open-world guiada por el error de predicción de un world-model
de video, con estabilidad de largo horizonte en cámara fija**. El veredicto es PIVOT
hacia ESE problema reducido. El proyecto grande, no; el problema chico adentro, quizás.

---

## 2. Contribución científica defendible

**Frase falsable (la única que sobrevive a un revisor):**
> El error de predicción de un world-model de video (V-JEPA 2.1) provee una señal de
> "sorpresa" temporal que detecta objetos/eventos novedosos (unknowns) en un stream de
> cámara fija con mejor recall y/o menor drift a 7 días que (a) self-training puro y
> (b) la misma arquitectura con un backbone estático superior (DINOv3), bajo idéntico
> presupuesto de cómputo y de supervisión.

Es falsable porque puede perder: si DINOv3 iguala o supera, la tesis cae. Esa es
exactamente la propiedad que la hace publicable (y la que el gate va a poner a prueba).

**Lo que NO se debe reclamar como novedad (lista negra):**

- "VLM verifica pseudo-labels para auto-entrenar un detector" — publicado 5+ veces:
  VLM-PL (2403.05346), SAS-Det (2308.06412), DST-Det (2310.01393), "The Detector
  Teaches Itself" (2605.03642), co-teaching VLM (2511.09955). Y empaquetado: Autodistill.
- "NL → destila un detector" — es literalmente el producto Autodistill.
- "Orquestar SAM2 + Grounding DINO + VLM en un pipeline" — SAM2Auto (2506.07850),
  ViperGPT/VisProg. Integración, no investigación.
- "Guardrails anti confirmation-bias con human-in-the-loop al 5%" — práctica estándar
  de self-training; un umbral en `LoopThresholds` no es contribución.
- "Analítica de personas en mall" — commodity de mercado. Cero novedad, y además es
  el framing que dispara el rechazo "esto es un producto, no un paper".
- "Detección densa con V-JEPA" sin más — DINOv3 ya lo supera en localización estática
  (ADE20K 47.9 vs 55.9; Cityscapes 73.5 vs 81.1; DAVIS 69.0 vs 71.1). Reclamarlo es
  perder de entrada.

La novedad vive en UNA cosa y solo una: **la señal temporal/predictiva como motor de
open-world en streams continuos**. Todo lo demás es andamiaje conocido.

---

## 3. Pregunta de investigación + hipótesis (reformuladas para publicar)

**RQ:** ¿Puede el error de predicción de un world-model de video servir como señal de
novedad open-world que (i) supere a backbones estáticos y a self-training puro en
recall de unknowns, y (ii) frene el drift de auto-entrenamiento en horizonte largo
(1/3/7 días) en cámara fija?

**H1 (temporal > estático):** En recall de unknowns, la señal de sorpresa de V-JEPA
2.1 supera a la misma señal derivada de DINOv3 por un margen ≥ X (definido en el gate).
*Si H1 falla, el world-model no aporta y el proyecto pierde su razón de existir.*

**H2 (circuit-breaker anti-drift):** Un VLM externo al EMA, actuando como circuit-
breaker disparado por la señal de novedad, reduce la degradación de mAP a 7 días
respecto de self-training puro, evitando colapso (definición de colapso en §4).

**H3 (no-redundancia):** La señal de sorpresa temporal es complementaria, no
redundante, con la incertidumbre del detector (correlación baja + ganancia aditiva en
ablation). *Esta es la que convierte "otra señal de uncertainty" en "señal nueva".*

Marketing prohibido: "intuición a nivel humano", "mejora casi solo", "auto-correctivo".
Esas frases del README hunden el paper. Reemplazar por afirmaciones medibles.

---

## 4. El GATE decisivo (2-3 semanas) — decide TODO

**Setup mínimo:** 1 cámara fija (mall o cualquier stream continuo público de horas),
el lazo de auto-corrección ya implementado, y la señal de novedad calculada de tres
formas sobre el MISMO pipeline y el MISMO presupuesto de cómputo/supervisión:

- **C1 — DINOv3** como fuente de la señal (backbone estático, baseline fuerte).
- **C2 — V-JEPA 2.1** error de predicción como señal de sorpresa temporal.
- **C3 — Híbrido** (DINOv3 espacial + V-JEPA temporal).

**Qué se mide (cuantitativo, sin ambigüedad):**

| Métrica | Definición |
|---|---|
| **Recall de unknowns** | objetos/eventos novedosos recuperados vs. ground-truth manual de un set de validación con unknowns inyectados o anotados. |
| **AUROC de la señal** | separabilidad known/unknown usando la señal como score. |
| **Complementariedad** | correlación de Pearson entre señal V-JEPA y uncertainty del detector (buscamos baja); y Δrecall al sumarlas. |
| **Costo** | latencia/FLOPs por frame de cada condición (para normalizar la comparación). |

**Criterio de ÉXITO (continuar el pivote):**
- C2 o C3 supera a C1 (DINOv3) en recall de unknowns por **≥ 5 puntos absolutos** (o
  AUROC **≥ +0.05**), **a paridad o menor costo**; **y**
- correlación señal-temporal vs. uncertainty del detector **< 0.5** (es complementaria);
- reproducible en **≥ 2** streams distintos (no overfit a una cámara).

**Criterio de FRACASO (matar el framing V-JEPA):**
- C1 (DINOv3) iguala o supera a C2/C3 dentro de **±2 puntos**; **o**
- la ventaja de V-JEPA desaparece al normalizar por costo; **o**
- la señal temporal correlaciona **> 0.7** con la uncertainty existente (es redundante).

**Si el gate FALLA:** NO-GO para el proyecto con V-JEPA. Dos salidas honestas:
(a) reescribir el paper sin world-model, como estudio de **estabilidad de largo
horizonte de self-training + VLM circuit-breaker** (más chico, sin mall, todavía
publicable en workshop si el anti-drift está bien medido); o (b) parar. No insistir
con el framing "world-model + mall": ese camino no llega a ningún lado, que es
exactamente lo que el usuario quiere evitar.

> El gate cuesta 2-3 semanas y puede ahorrar 6 meses. Es el experimento más barato que
> compra la mayor cantidad de certeza. **Hacerlo ANTES de tocar el resto del paper.**

---

## 5. Experimentos obligatorios para el paper (post-gate exitoso)

**Baselines (sin estos, rechazo automático):**
- Autodistill (NL→destila) sobre la misma cámara — el revisor lo va a pedir.
- SAS-Det (2308.06412) y/o DST-Det (2310.01393) — self-training OVD SOTA.
- AMROD (2406.16439) — el vecino más cercano en adaptación de detector en cámara.
- Self-training puro (sin VLM, sin señal de novedad) — el "lower bound" del lazo.

**Ablations (la columna vertebral del aporte):**
- self-training puro vs. **+ VLM circuit-breaker** vs. **+ señal de novedad V-JEPA**
  (las 3 condiciones del debate). Aditividad de cada componente.
- VLM disparado por novedad **vs.** VLM disparado siempre (justificar el trigger).
- circuit-breaker externo al EMA **vs.** dentro del EMA (mostrar por qué externo evita
  el feedback de error).
- sin guardrails vs. con guardrails (mostrar colapso sin ellos = motiva la pieza).

**Métrica/benchmark nuevo (esto es lo que un workshop premia):**
- **Curva de degradación a 1/3/7 días**: mAP (knowns) y recall (unknowns) vs. tiempo
  de operación continua. Definir **"colapso"** operacionalmente (p. ej. caída > 10% de
  mAP respecto al pico, o crecimiento monótono de FP por 3 ventanas seguidas).
- **Long-Horizon Stability Score**: una métrica escalar que combine retención de mAP +
  recall de unknowns + tasa de FP a 7 días. Proponerla nombrada es media contribución.
- Liberar el protocolo de evaluación de drift (split temporal, unknowns anotados). Un
  benchmark reproducible de drift en cámara fija con foundation models **no existe** y
  es defendible por sí solo.

---

## 6. Venue objetivo + fallback

**Objetivo (realista): workshop top-tier.**
- **Computer Vision in the Wild @ CVPR** o **Retail Vision @ECCV** o **AVSS**.
- Qué hace falta: gate aprobado + las 3 ablations + curva de drift 1/3/7 días +
  comparación contra Autodistill y un self-training-OVD SOTA. Con eso es competitivo.

**Stretch (difícil): main track (WACV es el más alcanzable; CVPR/ICCV/ECCV improbable).**
- Qué hace falta adicional: el benchmark/métrica nuevo adoptable por otros, ≥ 2-3
  datasets, evidencia de que la señal temporal de V-JEPA es **insustituible** (H1 con
  margen grande y robusto), y framing 100% "novelty detection / open-world", cero mall.
- WACV (Applications track) tolera mejor el ángulo aplicado; es el mejor tiro a main.

**Fallback (si el gate falla pero el anti-drift se sostiene):**
- Paper de **estabilidad de self-training a largo horizonte con VLM circuit-breaker**,
  sin world-model. Workshop de continual/open-world learning. Más chico pero real.

---

## 7. Qué CORTAR del alcance actual

Cortar agresivamente. Cada pieza de abajo agrega costo de integración y resta foco,
sin sumar a la contribución defendible:

- **El framing "mall / people-analytics".** Es commodity y dispara "esto es un producto".
  Usar mall solo como *uno* de los streams de prueba, nunca como el título ni la tesis.
- **La destilación a `distilled_detector.pt` y el split LOCAL/COLAB.** Es ingeniería de
  deployment; no aporta nada científico y consume semanas. Fuera del paper.
- **La web demo (Streamlit).** Demo de producto, no de investigación. Fuera del scope
  experimental.
- **SAM2 / segmentación.** No es parte de la tesis de novedad. Si no se mide segmentación
  como contribución, es overhead. Detección de cajas alcanza para el aporte.
- **"Intuición a nivel humano" / "mejora casi solo".** Borrar del README y de todo doc.
  Son red flags para revisores.
- **max_vlm_iterations / sobre-correccion como "feature".** Es un detalle de
  implementación, no un hallazgo. No venderlo como tal.

Lo que se CONSERVA: el lazo de auto-corrección como *infraestructura experimental*, el
VLM circuit-breaker, la señal de sorpresa de V-JEPA, el DriftMonitor convertido en
*instrumento de medición de drift*. Eso es todo lo que el paper necesita.

---

## 8. Probabilidad honesta de aceptación + riesgo residual

**Bajo el plan recomendado (PIVOT + gate aprobado):**

- **Workshop top-tier: 40-55%.** Aporte acotado, baselines correctos, métrica de drift
  nueva. Aceptable y realista.
- **WACV / main track aplicado: 15-25%.** Sube solo si H1 tiene margen grande y robusto
  y el benchmark es adoptable.
- **CVPR/ICCV/ECCV main track: < 10%.** No apostar a esto.

**Si se ignora el gate y se ejecuta el proyecto completo tal cual:** < 15% en cualquier
venio serio, con alto riesgo de 6 meses tirados. Ese es el escenario a evitar.

**Riesgo residual #1:** que la señal temporal de V-JEPA NO sea defendiblemente superior
a DINOv3 a paridad de costo. DINOv3 ya gana en localización estática; si gana también
en la señal de novedad, **no queda contribución**. Todo el veredicto cuelga de H1, y
por eso el gate existe: convierte ese riesgo de "se descubre en el mes 5 del review"
a "se descubre en la semana 3". El segundo riesgo es que el drift a largo horizonte sea
intratable y el circuit-breaker no lo frene de forma medible — pero eso, al menos, es
un resultado publicable aunque sea negativo, si se mide bien.

---

### Resumen para decisión

No ejecutar el proyecto completo. Reducirlo al problema de **novedad open-world por
error de predicción de world-model + estabilidad de largo horizonte**, correr el gate
de 2-3 semanas (V-JEPA vs DINOv3 vs híbrido sobre recall de unknowns y costo), y solo
seguir si V-JEPA/híbrido gana por ≥5 pts a paridad de costo y es complementario a la
uncertainty existente. Si gana: workshop top-tier con 40-55%. Si no: pivote al paper de
anti-drift sin world-model, o parar. Cortar mall, demo, destilación y "intuición humana".
