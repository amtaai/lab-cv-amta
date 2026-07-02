"""Web demo local (Streamlit).

Corre en la maquina del usuario y usa SOLO el modelo destilado bajado de
Colab a models/. No hace heavy lifting ni entrenamiento.

Ejecutar:
    streamlit run webdemo/app.py
"""

from __future__ import annotations

import streamlit as st

from core.config import load_config


def main() -> None:
    cfg = load_config()

    st.set_page_config(page_title="amta v-jepa-2 · detector intuitivo", layout="wide")
    st.title("amta v-jepa-2 — detector intuitivo (demo)")
    st.caption(
        "Demo local de inferencia. El entrenamiento y el lazo de auto-correccion "
        "corren en Colab; aca solo se usa el modelo destilado."
    )

    with st.sidebar:
        st.header("Estado")
        st.write(f"runtime: `{cfg.runtime.value}`")
        st.write(f"tier: `{cfg.tier.value}`")
        st.write(f"modelo destilado: `{cfg.distilled_model_path.name}`")
        if not cfg.distilled_model_path.exists():
            st.warning("Todavia no hay modelo destilado en models/. Entrenar en Colab y bajarlo.")

    prompts = st.text_input(
        "Que detectar (lenguaje natural)",
        value="una persona",
        help="La idea basica con la que arranca la camara.",
    )

    uploaded = st.file_uploader("Subi un video o imagen", type=["mp4", "mov", "jpg", "png"])
    if uploaded is None:
        st.info("Subi un archivo para correr la inferencia (placeholder).")
        return

    st.info("Inferencia pendiente de implementar (scaffold).")
    st.write({"prompts": prompts, "archivo": uploaded.name})


if __name__ == "__main__":
    main()
