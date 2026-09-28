"""Lectura/escritura del indice del corpus y del manifest de reproducibilidad.

metadata.json = el indice legible y diffeable (se versiona).
manifest.json = sha256 + tamano por clip (se versiona). Con los dos, el corpus
se reconstruye y se verifica sin git-lfs ni DVC.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from core.cascade.catalog.schema import ClipMeta

SCHEMA_VERSION = 1


def sha256_archivo(path: Path, chunk: int = 1 << 20) -> str:
    """sha256 en streaming (los videos no entran comodos en memoria)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(chunk), b""):
            h.update(bloque)
    return h.hexdigest()


def sondear_video(path: Path) -> dict:
    """Metadatos tecnicos via ffprobe: duracion, resolucion, fps, codec, n_frames."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,codec_name,nb_read_frames:format=duration",
        "-count_frames", "-of", "json", str(path),
    ]
    salida = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    datos = json.loads(salida)
    st = datos["streams"][0]
    num, den = (st.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps": round(fps, 3),
        "codec": st.get("codec_name", "unknown"),
        "n_frames": int(st.get("nb_read_frames") or 0),
        "duration_s": round(float(datos["format"]["duration"]), 3),
    }


def cargar_indice(corpus_dir: Path) -> list[ClipMeta]:
    """Lee metadata.json. Devuelve lista vacia si todavia no existe."""
    p = corpus_dir / "metadata.json"
    if not p.exists():
        return []
    datos = json.loads(p.read_text(encoding="utf-8"))
    return [ClipMeta.from_dict(c) for c in datos.get("clips", [])]


def guardar_indice(clips: list[ClipMeta], corpus_dir: Path) -> None:
    """Escribe metadata.json ordenado por clip_id (diffs estables)."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "n_clips": len(clips),
        "clips": [c.to_dict() for c in sorted(clips, key=lambda c: c.clip_id)],
    }
    (corpus_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def escribir_manifest(clips: list[ClipMeta], corpus_dir: Path) -> None:
    """Escribe manifest.json: sha256 + bytes por clip."""
    raw = corpus_dir / "raw"
    entradas = {}
    for c in sorted(clips, key=lambda c: c.clip_id):
        f = raw / c.filename
        entradas[c.clip_id] = {
            "filename": c.filename,
            "sha256": c.sha256,
            "size_bytes": f.stat().st_size if f.exists() else 0,
        }
    payload = {"schema_version": SCHEMA_VERSION, "n_clips": len(clips), "clips": entradas}
    (corpus_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def verificar_manifest(corpus_dir: Path) -> list[str]:
    """Devuelve los clip_id cuyo sha256 no coincide o cuyo archivo falta."""
    p = corpus_dir / "manifest.json"
    if not p.exists():
        return []
    datos = json.loads(p.read_text(encoding="utf-8"))
    raw = corpus_dir / "raw"
    malos = []
    for clip_id, e in datos.get("clips", {}).items():
        f = raw / e["filename"]
        if not f.exists() or sha256_archivo(f) != e["sha256"]:
            malos.append(clip_id)
    return sorted(malos)
