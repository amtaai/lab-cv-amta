"""Tests del esquema, indice y manifest del corpus."""

from __future__ import annotations

import json

from core.cascade.catalog.index import (
    cargar_indice,
    escribir_manifest,
    guardar_indice,
    sha256_archivo,
    verificar_manifest,
)
from core.cascade.catalog.schema import (
    CameraHeight,
    ClipMeta,
    CrowdDensity,
    Lighting,
    LocationType,
)


def _clip(tmp_path, nombre="a.mp4", contenido=b"video-bytes"):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / nombre).write_bytes(contenido)
    return ClipMeta(
        clip_id=nombre.replace(".mp4", ""),
        filename=nombre,
        source="test",
        camera_height=CameraHeight.CEILING,
        lighting=Lighting.BRIGHT,
        crowd_density=CrowdDensity.SPARSE,
        location_type=LocationType.STORE,
        duration_s=12.5,
        width=1920,
        height=1080,
        fps=25.0,
        codec="h264",
        n_frames=312,
        sha256=sha256_archivo(raw / nombre),
        license="CC-BY-4.0",
    )


def test_indice_roundtrip(tmp_path):
    clips = [_clip(tmp_path)]
    guardar_indice(clips, tmp_path)
    leidos = cargar_indice(tmp_path)
    assert len(leidos) == 1
    assert leidos[0].clip_id == "a"
    assert leidos[0].camera_height is CameraHeight.CEILING
    assert leidos[0].duration_s == 12.5


def test_manifest_detecta_archivo_corrupto(tmp_path):
    clips = [_clip(tmp_path)]
    guardar_indice(clips, tmp_path)
    escribir_manifest(clips, tmp_path)
    assert verificar_manifest(tmp_path) == []

    (tmp_path / "raw" / "a.mp4").write_bytes(b"otros-bytes")
    assert verificar_manifest(tmp_path) == ["a"]


def test_metadata_json_es_legible_y_diffeable(tmp_path):
    guardar_indice([_clip(tmp_path)], tmp_path)
    datos = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert datos["schema_version"] == 1
    assert datos["clips"][0]["camera_height"] == "ceiling"
