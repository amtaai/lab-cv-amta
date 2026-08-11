"""Catalogo del corpus: esquema, indice versionado y manifest de reproducibilidad."""

from __future__ import annotations

from core.cascade.catalog.index import (
    cargar_indice,
    escribir_manifest,
    guardar_indice,
    sha256_archivo,
    sondear_video,
    verificar_manifest,
)
from core.cascade.catalog.schema import (
    CameraHeight,
    ClipMeta,
    CrowdDensity,
    Lighting,
    LocationType,
)

__all__ = [
    "CameraHeight",
    "ClipMeta",
    "CrowdDensity",
    "Lighting",
    "LocationType",
    "cargar_indice",
    "escribir_manifest",
    "guardar_indice",
    "sha256_archivo",
    "sondear_video",
    "verificar_manifest",
]
