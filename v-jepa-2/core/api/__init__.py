"""API REST del cascade de vigilancia (FastAPI).

El pipeline no vive aca: esto lo expone. La deteccion esta en
`core/cascade/stage1_motion/`, `core/perception/` y `core/cascade/eval/`, y la
API solo traduce peticiones HTTP a llamadas del runner y respuestas del runner a
JSON validado.
"""
