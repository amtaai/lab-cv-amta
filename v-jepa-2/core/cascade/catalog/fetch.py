"""Descarga declarativa de clips de datasets publicos.

Politica: sources.json arranca VACIO a proposito. Antes de agregar una fuente hay
que verificar la licencia de redistribucion (ver README raiz: "manifest.psv por
dataset, con licencia verificada"). Nada se baja sin --accept-license explicito.

Uso:
  python -m core.cascade.catalog.fetch --list
  python -m core.cascade.catalog.fetch --source <nombre> --accept-license
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from core.cascade.config import load_cascade_config


def _cargar_fuentes(corpus_dir: Path) -> dict:
    p = corpus_dir / "sources.json"
    if not p.exists():
        return {"sources": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Baja clips de fuentes publicas registradas")
    ap.add_argument("--list", action="store_true", help="lista las fuentes registradas")
    ap.add_argument("--source", default=None)
    ap.add_argument("--accept-license", action="store_true",
                    help="confirma que la licencia de la fuente fue verificada")
    ap.add_argument("--dest", type=Path, default=None, help="por defecto: corpus/raw/_fetch")
    args = ap.parse_args()

    cfg = load_cascade_config()
    reg = _cargar_fuentes(cfg.corpus_dir)
    fuentes = reg.get("sources", {})

    if args.list or not args.source:
        if not fuentes:
            print("No hay fuentes registradas en corpus/sources.json.")
            print("Agregar una entrada {nombre: {url, license, files: [...]}} tras")
            print("verificar la licencia de redistribucion.")
            return 0
        for nombre, s in fuentes.items():
            print(f"{nombre:24s} {s.get('license','?'):16s} {len(s.get('files', []))} archivos")
        return 0

    if args.source not in fuentes:
        print(f"ABORTA: fuente desconocida: {args.source}")
        return 1
    if not args.accept_license:
        s = fuentes[args.source]
        print(f"ABORTA: {args.source} declara licencia '{s.get('license','?')}'.")
        print("Verificala y volve a correr con --accept-license.")
        return 1

    dest = args.dest or (cfg.raw_dir / "_fetch")
    dest.mkdir(parents=True, exist_ok=True)
    for url in fuentes[args.source].get("files", []):
        salida = dest / Path(url).name
        if salida.exists():
            print(f"skip (ya existe): {salida.name}")
            continue
        print(f"bajando {url} -> {salida}")
        urllib.request.urlretrieve(url, salida)
    print(f"OK. Ahora ingestar con: python -m core.cascade.catalog.ingest --file {dest}/<archivo>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
