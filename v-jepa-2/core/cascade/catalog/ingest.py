"""Ingesta de clips al corpus, de a uno (--file) o por lote (--dir).

Excepcion consciente a la convencion del repo (froth_gate no usa argparse):
la metadata es por-clip y no puede vivir en constantes de modulo.

--lighting es opcional: si no se pasa, se MIDE sobre los pixeles del clip
(ver catalog/medir.py) en vez de declararse a ojo. Las metricas crudas quedan
en ClipMeta.meta para poder reclasificar despues sin releer los videos.

Uso:
  python -m core.cascade.catalog.ingest --file /in/clip.mp4 --source ds \\
      --camera-height ceiling --crowd-density sparse --location-type store

  python -m core.cascade.catalog.ingest --dir /in/dataset --source ds \\
      --camera-height ceiling --crowd-density sparse --location-type store \\
      --notes "crowd_density SIN VERIFICAR"
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
from core.cascade.catalog.medir import medir_iluminacion
from core.cascade.catalog.schema import (
    CameraHeight,
    ClipMeta,
    CrowdDensity,
    Lighting,
    LocationType,
)
from core.cascade.config import load_cascade_config

MIN_LIBRE_GB = 5.0  # por debajo de esto no se ingesta
EXTENSIONES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _espacio_libre_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _ingestar_uno(origen: Path, clip_id: str, args, cfg, clips: list[ClipMeta]) -> ClipMeta | None:
    """Copia el clip a raw/, lo sondea, mide iluminacion y devuelve su ClipMeta."""
    destino = cfg.raw_dir / f"{clip_id}{origen.suffix}"
    if not destino.exists():
        shutil.copy2(origen, destino)

    try:
        tecnico = sondear_video(destino)
    except Exception as e:  # noqa: BLE001 - un clip corrupto no debe cortar el lote
        print(f"  SKIP {clip_id}: ffprobe fallo ({e})")
        destino.unlink(missing_ok=True)
        return None

    if args.lighting:
        lighting = Lighting(args.lighting)
        medicion = {"lighting_declarado": True}
    else:
        medicion = medir_iluminacion(destino)
        lighting = Lighting(medicion["lighting"])

    return ClipMeta(
        clip_id=clip_id,
        filename=destino.name,
        source=args.source,
        camera_height=CameraHeight(args.camera_height),
        lighting=lighting,
        crowd_density=CrowdDensity(args.crowd_density),
        location_type=LocationType(args.location_type),
        sha256=sha256_archivo(destino),
        license=args.license,
        notes=args.notes,
        meta=medicion,
        **tecnico,
    )


def _remedir(cfg) -> int:
    """Re-mide la iluminacion de los clips ya indexados y reescribe el indice.

    Existe para poder cambiar la regla de clasificacion (o las metricas) sin
    volver a copiar 182 archivos al corpus.
    """
    clips = cargar_indice(cfg.corpus_dir)
    if not clips:
        print("ABORTA: corpus vacio, no hay nada que re-medir.")
        return 1

    cambios, reparto = 0, {}
    for i, c in enumerate(clips, 1):
        ruta = cfg.raw_dir / c.filename
        if not ruta.exists():
            print(f"  SKIP {c.clip_id}: falta {ruta}")
            continue
        medicion = medir_iluminacion(ruta)
        antes = c.lighting
        c.lighting = Lighting(medicion["lighting"])
        c.meta = {**c.meta, **medicion}
        if c.lighting is not antes:
            cambios += 1
        reparto[c.lighting.value] = reparto.get(c.lighting.value, 0) + 1
        if i % 40 == 0 or i == len(clips):
            print(f"  {i}/{len(clips)} re-medidos", flush=True)

    guardar_indice(clips, cfg.corpus_dir)
    escribir_manifest(clips, cfg.corpus_dir)
    print(f"OK: {len(clips)} clips re-medidos, {cambios} cambiaron de etiqueta.")
    print(f"iluminacion: {reparto}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Agrega clips al corpus")
    grupo = ap.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--file", type=Path, help="un solo clip")
    grupo.add_argument("--dir", type=Path, help="directorio con clips (recursivo)")
    grupo.add_argument("--remedir", action="store_true",
                       help="re-mide la iluminacion de los clips ya indexados y "
                            "reescribe metadata.json (no copia nada)")
    ap.add_argument("--clip-id", default=None,
                    help="solo con --file; por defecto el nombre sin extension")
    ap.add_argument("--prefix", default="", help="prefijo para los clip_id del lote")
    # No van como required=True porque --remedir no los necesita; se validan abajo.
    ap.add_argument("--source")
    ap.add_argument("--camera-height", choices=[e.value for e in CameraHeight])
    ap.add_argument("--lighting", default=None, choices=[e.value for e in Lighting],
                    help="si se omite, se mide sobre los pixeles del clip")
    ap.add_argument("--crowd-density", choices=[e.value for e in CrowdDensity])
    ap.add_argument("--location-type", choices=[e.value for e in LocationType])
    ap.add_argument("--license", default="unknown")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cfg = load_cascade_config()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    if args.remedir:
        return _remedir(cfg)

    faltan = [n for n, v in (("--source", args.source),
                             ("--camera-height", args.camera_height),
                             ("--crowd-density", args.crowd_density),
                             ("--location-type", args.location_type)) if not v]
    if faltan:
        print(f"ABORTA: faltan argumentos obligatorios: {', '.join(faltan)}")
        return 2

    libre = _espacio_libre_gb(cfg.raw_dir)
    if libre < MIN_LIBRE_GB:
        print(f"ABORTA: solo {libre:.1f} GB libres (minimo {MIN_LIBRE_GB} GB)")
        return 1

    if args.file:
        if not args.file.exists():
            print(f"ABORTA: no existe {args.file}")
            return 1
        pendientes = [(args.file, args.clip_id or args.file.stem)]
    else:
        if not args.dir.is_dir():
            print(f"ABORTA: no es un directorio: {args.dir}")
            return 1
        encontrados = sorted(
            p for p in args.dir.rglob("*") if p.suffix.lower() in EXTENSIONES
        )
        if not encontrados:
            print(f"ABORTA: no hay videos {EXTENSIONES} en {args.dir}")
            return 1
        # El clip_id lleva el subdirectorio para no colisionar entre carpetas
        # (shoplifting/clip-1.mp4 y normal/clip-1.mp4 son clips distintos).
        pendientes = []
        for p in encontrados:
            rel = p.relative_to(args.dir).with_suffix("")
            pendientes.append((p, args.prefix + str(rel).replace("/", "_")))

    clips = cargar_indice(cfg.corpus_dir)
    existentes = {c.clip_id for c in clips}
    nuevos, saltados = 0, 0

    for i, (origen, clip_id) in enumerate(pendientes, 1):
        if clip_id in existentes:
            saltados += 1
            continue
        clip = _ingestar_uno(origen, clip_id, args, cfg, clips)
        if clip is None:
            continue
        clips.append(clip)
        existentes.add(clip_id)
        nuevos += 1
        if len(pendientes) == 1 or i % 20 == 0 or i == len(pendientes):
            print(f"  {i}/{len(pendientes)} {clip_id}: {clip.duration_s}s "
                  f"{clip.width}x{clip.height} @{clip.fps}fps {clip.codec} "
                  f"{clip.n_frames}f luz={clip.lighting.value}", flush=True)

    if not nuevos:
        print(f"Nada nuevo que ingestar ({saltados} ya estaban en el indice).")
        return 0

    guardar_indice(clips, cfg.corpus_dir)
    escribir_manifest(clips, cfg.corpus_dir)

    total_gb = sum(
        (cfg.raw_dir / c.filename).stat().st_size
        for c in clips
        if (cfg.raw_dir / c.filename).exists()
    ) / 1e9
    reparto: dict[str, int] = {}
    for c in clips:
        reparto[c.lighting.value] = reparto.get(c.lighting.value, 0) + 1
    print(f"OK: {nuevos} clips nuevos, {saltados} ya estaban.")
    print(f"corpus: {len(clips)} clips | {sum(c.n_frames for c in clips)} frames | "
          f"{total_gb:.2f} GB | {_espacio_libre_gb(cfg.raw_dir):.1f} GB libres")
    print(f"iluminacion medida: {reparto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
