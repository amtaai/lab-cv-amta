"""Ingesta de clips al corpus.

Excepcion consciente a la convencion del repo (froth_gate no usa argparse):
la metadata es por-clip y no puede vivir en constantes de modulo.

Uso:
  python -m core.cascade.catalog.ingest --file /clips/x.mp4 --source ieee \\
      --camera-height ceiling --lighting bright --crowd-density sparse \\
      --location-type store --license CC-BY-4.0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from core.cascade.catalog.index import (
    cargar_indice,
    escribir_manifest,
    guardar_indice,
    sha256_archivo,
    sondear_video,
)
from core.cascade.catalog.schema import (
    CameraHeight,
    ClipMeta,
    CrowdDensity,
    Lighting,
    LocationType,
)
from core.cascade.config import load_cascade_config

MIN_LIBRE_GB = 5.0  # por debajo de esto no se ingesta (host al 88% de uso)


def _espacio_libre_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description="Agrega un clip al corpus")
    ap.add_argument("--file", required=True, type=Path)
    ap.add_argument("--clip-id", default=None, help="por defecto: nombre del archivo sin extension")
    ap.add_argument("--source", required=True)
    ap.add_argument("--camera-height", required=True, choices=[e.value for e in CameraHeight])
    ap.add_argument("--lighting", required=True, choices=[e.value for e in Lighting])
    ap.add_argument("--crowd-density", required=True, choices=[e.value for e in CrowdDensity])
    ap.add_argument("--location-type", required=True, choices=[e.value for e in LocationType])
    ap.add_argument("--license", default="unknown")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cfg = load_cascade_config()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    libre = _espacio_libre_gb(cfg.raw_dir)
    if libre < MIN_LIBRE_GB:
        print(f"ABORTA: solo {libre:.1f} GB libres (minimo {MIN_LIBRE_GB} GB)")
        return 1

    origen = args.file
    if not origen.exists():
        print(f"ABORTA: no existe {origen}")
        return 1

    clip_id = args.clip_id or origen.stem
    clips = cargar_indice(cfg.corpus_dir)
    if any(c.clip_id == clip_id for c in clips):
        print(f"ABORTA: clip_id ya existe en el indice: {clip_id}")
        return 1

    destino = cfg.raw_dir / f"{clip_id}{origen.suffix}"
    if not destino.exists():
        shutil.copy2(origen, destino)

    tecnico = sondear_video(destino)
    clip = ClipMeta(
        clip_id=clip_id,
        filename=destino.name,
        source=args.source,
        camera_height=CameraHeight(args.camera_height),
        lighting=Lighting(args.lighting),
        crowd_density=CrowdDensity(args.crowd_density),
        location_type=LocationType(args.location_type),
        sha256=sha256_archivo(destino),
        license=args.license,
        notes=args.notes,
        **tecnico,
    )
    clips.append(clip)
    guardar_indice(clips, cfg.corpus_dir)
    escribir_manifest(clips, cfg.corpus_dir)

    total_gb = sum(
        (cfg.raw_dir / c.filename).stat().st_size
        for c in clips
        if (cfg.raw_dir / c.filename).exists()
    ) / 1e9
    print(f"OK {clip_id}: {tecnico['duration_s']}s {tecnico['width']}x{tecnico['height']} "
          f"@{tecnico['fps']}fps {tecnico['codec']} ({tecnico['n_frames']} frames)")
    print(f"corpus: {len(clips)} clips | {total_gb:.2f} GB | {libre:.1f} GB libres")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
