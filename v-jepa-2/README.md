# amta v-jepa-2 — detector de objetos intuitivo y auto-correctivo

Sistema que combina **V-JEPA 2** (world model de video) + **segmentacion**
(SAM2 + open-vocabulary) + **VQA** para lograr deteccion de objetos "intuitiva
a nivel humano". Una camara arranca con una idea muy basica de que detectar
(expresada en lenguaje natural) y el sistema **se auto-corrige** hasta detectar
de forma casi autonoma, sin re-etiquetado humano constante.

Caso de uso de referencia: analizar la gente que pasa por una zona de un mall
durante X horas.

> La novedad no es ningun componente suelto (todos son estado del arte), sino
> la **orquestacion integrada con un lazo de auto-correccion robusto**.

## Topologia

- **Local** (esta maquina): web demo + inferencia del **modelo destilado**. Sin entrenamiento.
- **Colab** (GPU 80GB / 120GB RAM): heavy lifting — V-JEPA2, SAM2, detector
  open-vocab, VLM verificador, pseudo-labeling y fine-tuning. Exporta el modelo
  destilado que se baja a `models/`.

## Arrancar (local)

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run webdemo/app.py
```

El pipeline pesado se corre desde `colab/train_pipeline.ipynb` en Colab.

## Estructura

- `core/` — paquete de integracion (percepcion, razonamiento, lazo, orquestador)
- `webdemo/` — demo local (Streamlit)
- `colab/` — notebook de entrenamiento (Colab)
- `third_party/vjepa2/` — repo upstream de V-JEPA 2 (no editar)

Ver `CLAUDE.md` para la arquitectura completa y los detalles del lazo.
