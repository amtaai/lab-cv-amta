# 02 - Arquitectura del detector intuitivo auto-correctivo

> Diagramas extensivos (Mermaid + ASCII) de la arquitectura del proyecto:
> V-JEPA 2/2.1 + SAM2 + detector open-vocabulary + VLM verificador, dentro de un
> lazo de auto-correccion que genera pseudo-labels verificados, reentrena y destila
> un detector. Caso de referencia: analitica de personas en un mall, camara fija.
>
> Todas las etiquetas dentro de los nodos Mermaid van **sin tildes** (para evitar
> problemas de render). El texto explicativo de cada diagrama si lleva tildes.
> Los nombres de clases, modulos, tipos y umbrales son **identicos a los del codigo**
> (`core/`), no inventados.

## Indice de diagramas

1. [Vista de sistema: topologia LOCAL vs COLAB](#1-vista-de-sistema-topologia-local-vs-colab)
2. [Pipeline de percepcion por frame](#2-pipeline-de-percepcion-por-frame)
3. [El lazo de auto-correccion (ciclo central)](#3-el-lazo-de-auto-correccion-ciclo-central)
4. [Diagrama de secuencia de una iteracion del lazo](#4-diagrama-de-secuencia-de-una-iteracion-del-lazo)
5. [Ciclo de vida de entrenamiento, destilacion y promocion del modelo](#5-ciclo-de-vida-de-entrenamiento-destilacion-y-promocion-del-modelo)
6. [Maquina de estados de una deteccion](#6-maquina-de-estados-de-una-deteccion)
7. [Mapa de guardrails y umbrales de LoopThresholds](#7-mapa-de-guardrails-y-umbrales-de-loopthresholds)

---

## 1. Vista de sistema: topologia LOCAL vs COLAB

```mermaid
flowchart TB
    subgraph LOCAL["LOCAL (Windows, sin GPU de entrenamiento)"]
        direction TB
        WEB["webdemo/app.py<br/>Streamlit web demo"]
        ORQ_L["IntuitiveDetector._process_local<br/>(inferencia liviana)"]
        DIST["models/distilled_detector.pt<br/>(checkpoint destilado, gitignored)"]
        WEB --> ORQ_L
        ORQ_L --> DIST
        DIST -.->|FrameResult| ORQ_L
    end

    subgraph COLAB["COLAB (GPU 80GB VRAM / 120GB RAM)"]
        direction TB
        NB["colab/train_pipeline.ipynb<br/>(setea AMTA_RUNTIME=colab)"]
        ORQ_C["IntuitiveDetector._process_full_loop<br/>(lazo completo)"]
        PERC["Percepcion pesada<br/>detector + Sam2Segmenter + Tracker + VJepa2Encoder"]
        REAS["Razonamiento<br/>VlmVerifier"]
        LOOP["Lazo<br/>PseudoLabeler + DriftMonitor"]
        TRAIN["Trainer<br/>fine_tune + distill"]
        NB --> ORQ_C
        ORQ_C --> PERC --> REAS --> LOOP --> TRAIN
    end

    CFG["core/config.py<br/>Runtime(local|colab) + Tier(server|edge)<br/>via AMTA_RUNTIME / AMTA_TIER"]
    CFG -.->|bifurca runtime| ORQ_L
    CFG -.->|bifurca runtime| ORQ_C

    TRAIN ==>|"CONTRATO: produce distilled_detector.pt"| DIST

    classDef local fill:#e8f4ff,stroke:#2b6cb0,color:#1a365d;
    classDef colab fill:#fff5e6,stroke:#dd6b20,color:#7b341e;
    classDef cfg fill:#f0fff4,stroke:#2f855a,color:#22543d;
    class WEB,ORQ_L,DIST local;
    class NB,ORQ_C,PERC,REAS,LOOP,TRAIN colab;
    class CFG cfg;
```

**Explicacion.** El sistema tiene dos lados con responsabilidades estrictamente
separadas. En **LOCAL** (la maquina Windows del usuario, sin GPU de entrenamiento)
solo corre la web demo (`webdemo/app.py`, Streamlit) y la **inferencia liviana**
del modelo ya destilado, a traves de `IntuitiveDetector._process_local`, que carga
`models/distilled_detector.pt`. En **COLAB** (GPU de 80GB) ocurre todo el heavy
lifting: el pipeline de percepcion pesado, el VLM verificador, el lazo de
pseudo-labeling/guardrails y el entrenamiento/destilacion, orquestado por
`IntuitiveDetector._process_full_loop`. El switch lo determina `core/config.py`
leyendo las variables de entorno `AMTA_RUNTIME` (`local`|`colab`) y `AMTA_TIER`
(`server`|`edge`). El **contrato** que une ambos lados es unidireccional: Colab
entrena y destila, produce `distilled_detector.pt`, y ese unico artefacto **baja a
`models/`** para que el lado local lo consuma. Nunca se cargan V-JEPA2/SAM2/VLM ni
se entrena en local (de hecho `Trainer.__init__` lanza `RuntimeError` si se
instancia en runtime local).

---

## 2. Pipeline de percepcion por frame

```mermaid
flowchart LR
    FRAME["frame: np.ndarray<br/>(HxWx3 RGB)"]
    PROMPTS["prompts: list[str]<br/>('una persona', ...)"]

    subgraph PERCEPCION["Percepcion por frame (core/perception)"]
        direction LR
        DET["OpenVocabDetector.detect<br/>GroundingDinoDetector / YoloWorldDetector"]
        SAM["Sam2Segmenter<br/>init_video / add_prompts / propagate"]
        TRK["Tracker.update<br/>(ByteTrack / FairMOT)"]
        ENC["VJepa2Encoder.extract_features<br/>(backbone CONGELADO)"]
    end

    FRAME --> DET
    PROMPTS --> DET
    DET -->|"list[Detection]<br/>bbox + label + score"| SAM
    SAM -->|"list[Detection]<br/>+ mask (HxW)"| TRK
    TRK -->|"list[Detection]<br/>+ track_id"| OUT
    FRAME --> ENC
    ENC -->|"features (T,H,W,3) -> embeddings<br/>semantica / novedad / re-ID"| TRK

    OUT["FrameResult<br/>frame_index + detections[]"]

    classDef io fill:#f7fafc,stroke:#4a5568,color:#1a202c;
    classDef comp fill:#ebf8ff,stroke:#3182ce,color:#1a365d;
    class FRAME,PROMPTS,OUT io;
    class DET,SAM,TRK,ENC comp;
```

**Explicacion.** Este es el camino que recorre **un frame** en el lado pesado.
Primero el **detector open-vocabulary** (`OpenVocabDetector.detect`, implementado
por `GroundingDinoDetector` o `YoloWorldDetector`) propone cajas a partir de los
`prompts` en lenguaje natural; su salida es una `list[Detection]` con `bbox`,
`label` y `score`. Esas detecciones pasan a **`Sam2Segmenter`**, que con `add_prompts`
refina cada caja a una **mascara binaria** (`Detection.mask`) y con `init_video` /
`propagate` mantiene **memoria temporal** para arrastrar mascaras frame a frame.
Luego el **`Tracker.update`** (ByteTrack/FairMOT) asocia cada deteccion con su
trayectoria y le asigna un **`track_id`** estable, lo que permite contar instancias
unicas (`Tracker.unique_count`). En paralelo, el **`VJepa2Encoder`** (backbone
**congelado** de V-JEPA 2.1) extrae embeddings espaciotemporales del clip; no produce
cajas ni mascaras, sino senales de semantica, novedad y re-identificacion que
alimentan al tracker y a etapas downstream. El agregado de todo es un `FrameResult`
con la lista de `Detection` enriquecidas.

---

## 3. El lazo de auto-correccion (ciclo central)

```mermaid
flowchart TB
    PERC["Percepcion por frame<br/>(diagrama 2) -> list[Detection]"]
    VLM["VlmVerifier.verify_batch<br/>Florence-2 / InternVL<br/>marca verified / corrige label"]
    FILT{"Filtro de confianza<br/>score >= pseudo_label_min_score<br/>AND verified == True ?"}
    PL["PseudoLabeler.emit<br/>persiste pseudo-labels al store (data/)"]
    DM["DriftMonitor<br/>observe / is_drifting<br/>sample_for_human_review"]
    DATASET[("data/pseudo_labels<br/>dataset auto-etiquetado")]
    TRAIN["Trainer.fine_tune + distill<br/>(COLAB)"]
    MODEL["models/distilled_detector.pt"]
    HUMAN["Revision humana<br/>(human-in-the-loop)"]

    PERC --> VLM
    VLM --> FILT
    FILT -->|"acepta"| PL
    FILT -->|"rechaza"| DROP["descartar deteccion"]
    PL --> DM
    DM -->|"muestra (human_review_fraction)"| HUMAN
    DM -->|"drift detectado: alerta / pausa"| HUMAN
    HUMAN -.->|correcciones| DATASET
    PL --> DATASET
    DATASET --> TRAIN
    TRAIN --> MODEL
    MODEL -.->|"detector mejorado realimenta la percepcion"| PERC

    classDef loop fill:#fef6f6,stroke:#c53030,color:#742a2a;
    classDef store fill:#f0fff4,stroke:#2f855a,color:#22543d;
    classDef guard fill:#fffbea,stroke:#b7791f,color:#744210;
    class PERC,VLM,PL,TRAIN,MODEL loop;
    class DATASET store;
    class FILT,DM,HUMAN guard;
```

**Explicacion.** Este es el corazon del proyecto: el **lazo de auto-correccion**.
Las detecciones que salen de la percepcion entran al **`VlmVerifier.verify_batch`**,
que actua como verificador externo (los VLM son malos auto-corrigiendose pero buenos
verificando): por cada deteccion dudosa decide si la acepta, la descarta o **corrige
su `label`**, marcando `Detection.verified`. Luego un **filtro de confianza** deja
pasar solo lo que cumple `score >= pseudo_label_min_score` **y** esta verificado (la
logica vive en `PseudoLabeler.is_trustworthy`). Lo que pasa se convierte en
**pseudo-label** via `PseudoLabeler.emit`, que escribe al store `data/pseudo_labels`.
El **`DriftMonitor`** vigila la salud del lazo: registra confianzas (`observe`),
detecta degradacion (`is_drifting`) y **samplea un porcentaje** de detecciones para
revision humana (`sample_for_human_review`), evitando el **confirmation bias** (que el
modelo refuerce sus propios errores). El dataset auto-etiquetado alimenta al
**`Trainer`** (en Colab), que afina y destila el modelo; el `distilled_detector.pt`
resultante mejora la percepcion y cierra el **ciclo**.

---

## 4. Diagrama de secuencia de una iteracion del lazo

```mermaid
sequenceDiagram
    autonumber
    participant ORQ as IntuitiveDetector
    participant DET as OpenVocabDetector
    participant SAM as Sam2Segmenter
    participant TRK as Tracker
    participant ENC as VJepa2Encoder
    participant VLM as VlmVerifier
    participant PL as PseudoLabeler
    participant DM as DriftMonitor
    participant TR as Trainer

    ORQ->>DET: detect(frame, prompts)
    DET-->>ORQ: list[Detection] (bbox, label, score)
    ORQ->>SAM: add_prompts(detections) / propagate(frame)
    SAM-->>ORQ: list[Detection] + mask
    ORQ->>ENC: extract_features(clip)
    ENC-->>ORQ: embeddings (novedad / re-ID)
    ORQ->>TRK: update(detections)
    TRK-->>ORQ: list[Detection] + track_id

    loop hasta max_vlm_iterations
        ORQ->>VLM: verify_batch(frame, detections)
        VLM-->>ORQ: Verdict(accepted, corrected_label, agreement)
    end

    ORQ->>PL: emit(frame_index, detections)
    Note over PL: solo si verified AND<br/>score >= pseudo_label_min_score
    PL-->>ORQ: n pseudo-labels emitidos

    ORQ->>DM: observe(detections)
    ORQ->>DM: sample_for_human_review(detections)
    DM-->>ORQ: subset a revision humana

    alt periodicamente (en Colab)
        ORQ->>TR: fine_tune(dataset_dir)
        TR-->>ORQ: teacher_ckpt
        ORQ->>TR: distill(teacher_ckpt)
        TR-->>ORQ: distilled_detector.pt
    end
```

**Explicacion.** La secuencia muestra **una iteracion completa** del lazo en Colab,
orquestada por `IntuitiveDetector._process_full_loop`. Primero la percepcion:
el `OpenVocabDetector` propone cajas, `Sam2Segmenter` agrega mascaras con memoria
temporal, `VJepa2Encoder` aporta embeddings y `Tracker` asigna `track_id`. Despues
viene la verificacion: el **`VlmVerifier`** se llama dentro de un bucle acotado a
**`max_vlm_iterations`** (mas de ~3 iteraciones tiende a sobre-corregir), devolviendo
un **`Verdict`** con `accepted`, `corrected_label` y `agreement`. El `PseudoLabeler`
emite labels solo para lo verificado y confiable. El `DriftMonitor` observa y samplea
para revision humana. Finalmente, **de forma periodica** (no en cada frame), el
`Trainer` ejecuta `fine_tune` y luego `distill`, produciendo el checkpoint compacto
que baja a local.

---

## 5. Ciclo de vida de entrenamiento, destilacion y promocion del modelo

```mermaid
flowchart TB
    subgraph FASE1["Fase 1 - Bootstrap (COLAB)"]
        P1["Prompts en lenguaje natural<br/>+ clips del mall"]
        P2["Percepcion pesada<br/>detector + SAM2 + tracker + V-JEPA2"]
        P1 --> P2
    end

    subgraph FASE2["Fase 2 - Auto-etiquetado verificado (COLAB)"]
        P3["VlmVerifier: verifica / corrige"]
        P4["PseudoLabeler: filtra por umbrales y emite"]
        P5["DriftMonitor: guardrails + human-in-the-loop"]
        P3 --> P4 --> P5
    end

    subgraph FASE3["Fase 3 - Entrenamiento + destilacion (COLAB)"]
        P6[("data/pseudo_labels<br/>dataset acumulado")]
        P7["Trainer.fine_tune<br/>-> teacher target"]
        P8["Trainer.distill<br/>-> modelo compacto"]
        P6 --> P7 --> P8
    end

    subgraph FASE4["Fase 4 - Promocion a LOCAL"]
        P9["distilled_detector.pt"]
        P10["models/ (gitignored, baja de Colab)"]
        P11["IntuitiveDetector._process_local<br/>inferencia en la web demo"]
        P9 --> P10 --> P11
    end

    P2 --> P3
    P5 --> P6
    P8 --> P9
    P11 -.->|"nuevos clips realimentan el bootstrap"| P1

    classDef f1 fill:#ebf8ff,stroke:#3182ce,color:#1a365d;
    classDef f2 fill:#fffbea,stroke:#b7791f,color:#744210;
    classDef f3 fill:#fff5f5,stroke:#c53030,color:#742a2a;
    classDef f4 fill:#f0fff4,stroke:#2f855a,color:#22543d;
    class P1,P2 f1;
    class P3,P4,P5 f2;
    class P6,P7,P8 f3;
    class P9,P10,P11 f4;
```

**Explicacion.** El ciclo de vida del modelo atraviesa cuatro fases. En la **Fase 1
(Bootstrap)** se arranca solo con prompts en lenguaje natural y clips del mall, y la
percepcion pesada produce detecciones iniciales sin clases pre-entrenadas. En la
**Fase 2 (Auto-etiquetado verificado)** el `VlmVerifier` corrige, el `PseudoLabeler`
filtra por los umbrales y emite pseudo-labels, y el `DriftMonitor` aplica los
guardrails y separa muestras para revision humana. En la **Fase 3 (Entrenamiento +
destilacion)** el `Trainer` consume el dataset acumulado en `data/pseudo_labels`:
`fine_tune` produce un teacher target y `distill` lo comprime a un modelo compacto.
En la **Fase 4 (Promocion)** el `distilled_detector.pt` baja a `models/` y queda
disponible para `IntuitiveDetector._process_local`, que lo usa en la web demo. Los
nuevos clips que se ven en produccion realimentan el bootstrap, haciendo el sistema
iterativo y cada vez mas autonomo.

---

## 6. Maquina de estados de una deteccion

```mermaid
stateDiagram-v2
    [*] --> Propuesta: detector.detect()

    Propuesta --> Descartada_por_score: score < detector_min_score
    Propuesta --> Segmentada: score >= detector_min_score

    Segmentada --> Trackeada: Tracker.update() asigna track_id
    Trackeada --> EnVerificacion: enviada al VlmVerifier

    state EnVerificacion {
        [*] --> Iterando
        Iterando --> Iterando: iter < max_vlm_iterations
        Iterando --> [*]: corte por max_vlm_iterations
    }

    EnVerificacion --> Verificada: agreement >= vlm_verify_min_agreement
    EnVerificacion --> Corregida: VLM cambia label (corrected_label)
    EnVerificacion --> Rechazada: agreement < vlm_verify_min_agreement

    Corregida --> Verificada: re-evaluada con nuevo label

    Verificada --> Aceptada: verified == True
    Aceptada --> PseudoLabel: score >= pseudo_label_min_score
    Aceptada --> Descartada_por_score: score < pseudo_label_min_score

    PseudoLabel --> AEntrenamiento: persistida en data/pseudo_labels
    PseudoLabel --> ARevisionHumana: muestreada (human_review_fraction)

    ARevisionHumana --> AEntrenamiento: corregida por humano
    ARevisionHumana --> Rechazada: descartada por humano

    Rechazada --> [*]
    Descartada_por_score --> [*]
    AEntrenamiento --> [*]
```

**Explicacion.** Esta maquina de estados sigue el ciclo de vida de **una sola
deteccion**. Nace como **Propuesta** del detector open-vocab; si su `score` no supera
`detector_min_score` se descarta de inmediato. Si sobrevive, pasa por **Segmentada**
(SAM2) y **Trackeada** (recibe `track_id`), y entra a **EnVerificacion**, un
sub-estado donde el `VlmVerifier` itera **acotado a `max_vlm_iterations`**. Segun el
`Verdict`, la deteccion queda **Verificada** (acuerdo suficiente), **Corregida** (el
VLM le cambia el `label` y se re-evalua) o **Rechazada** (acuerdo por debajo de
`vlm_verify_min_agreement`). Una deteccion verificada se vuelve **Aceptada**, y solo
si su score supera `pseudo_label_min_score` se promueve a **PseudoLabel**. Desde ahi,
o va directo **AEntrenamiento** (persistida en `data/pseudo_labels`), o si fue
muestreada por `human_review_fraction` pasa **ARevisionHumana**, donde un humano la
corrige (y entonces va a entrenamiento) o la descarta.

---

## 7. Mapa de guardrails y umbrales de LoopThresholds

```mermaid
flowchart TB
    subgraph THRESH["core/config.py :: LoopThresholds"]
        T1["detector_min_score = 0.30"]
        T2["pseudo_label_min_score = 0.70"]
        T3["vlm_verify_min_agreement = 0.60"]
        T4["max_vlm_iterations = 3"]
        T5["human_review_fraction = 0.05"]
    end

    G1["GUARDRAIL 1<br/>Filtro de entrada del detector"]
    G2["GUARDRAIL 2<br/>Corte de iteraciones del VLM"]
    G3["GUARDRAIL 3<br/>Acuerdo minimo del VLM"]
    G4["GUARDRAIL 4<br/>Calidad minima para pseudo-label"]
    G5["GUARDRAIL 5<br/>Muestreo human-in-the-loop"]

    C1["OpenVocabDetector.detect<br/>(box_threshold / conf ~ detector_min_score)"]
    C2["VlmVerifier.verify_batch<br/>(loop acotado)"]
    C3["VlmVerifier -> Verdict.agreement"]
    C4["PseudoLabeler.is_trustworthy"]
    C5["DriftMonitor.sample_for_human_review<br/>+ is_drifting"]

    T1 --> G1 --> C1
    T4 --> G2 --> C2
    T3 --> G3 --> C3
    T2 --> G4 --> C4
    T5 --> G5 --> C5

    C1 -->|detecciones validas| FLOW(("LAZO"))
    C2 --> FLOW
    C3 --> FLOW
    C4 --> FLOW
    C5 --> FLOW

    classDef th fill:#fffbea,stroke:#b7791f,color:#744210;
    classDef gd fill:#fff5f5,stroke:#c53030,color:#742a2a;
    classDef cd fill:#ebf8ff,stroke:#3182ce,color:#1a365d;
    class T1,T2,T3,T4,T5 th;
    class G1,G2,G3,G4,G5 gd;
    class C1,C2,C3,C4,C5 cd;
```

**Explicacion.** El proyecto centraliza todos sus guardrails anti confirmation-bias
en `core/config.py::LoopThresholds`, y cada umbral se traduce en un punto concreto de
control dentro del lazo. **`detector_min_score` (0.30)** filtra la entrada del
detector open-vocab, descartando propuestas debiles (se corresponde con el
`box_threshold`/`conf` de `GroundingDinoDetector`/`YoloWorldDetector`).
**`max_vlm_iterations` (3)** acota el bucle del `VlmVerifier` para evitar la
sobre-correccion (mas de ~3 pasadas degrada el resultado). **`vlm_verify_min_agreement`
(0.60)** define el acuerdo minimo del VLM para aceptar un `Verdict`.
**`pseudo_label_min_score` (0.70)** es el corte de calidad que aplica
`PseudoLabeler.is_trustworthy` antes de convertir una deteccion en pseudo-label
(solo lo muy confiable y verificado entra al dataset). **`human_review_fraction`
(0.05)** gobierna cuanto muestrea el `DriftMonitor` para revision humana, el seguro
ultimo contra que el lazo refuerce sus propios errores. La regla del proyecto es
inviolable: cualquier cambio al lazo debe respetar estos umbrales y leerlos siempre
desde `core/config.py`, nunca hardcodeados.

---

### Referencias de componentes (ver `papers.md`, `CLAUDE.md` y [`03_estado_del_arte.md`](03_estado_del_arte.md))

- **V-JEPA 2** (arxiv 2506.09985) y **V-JEPA 2.1** (arxiv 2603.14482, verificado 2026-06-17) -> `VJepa2Encoder`
- **SAM 2** (arxiv 2408.00714) -> `Sam2Segmenter`
- **Grounding DINO** (arxiv 2303.05499) y **YOLO-World** (arxiv 2401.17270) -> `OpenVocabDetector`
- **Florence-2** / **InternVL** -> `VlmVerifier`
- **ByteTrack** / **FairMOT** -> `Tracker`
- **Mask2Former** (arxiv 2112.01527), **Autodistill** -> referencias de destilacion/segmentacion
