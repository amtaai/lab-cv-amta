"""Exporta los resultados a CSV.

Dos archivos, porque son dos granularidades distintas y mezclarlas da una tabla
que no sirve para ninguna de las dos cosas:

- `detecciones.csv`: una fila por caja detectada. Sirve para auditar el
  seguimiento a mano, para reconstruir trayectorias o para cruzarlo con otra
  herramienta.
- `resumen_clips.csv`: una fila por clip, con conteo, tiempos y costes. Es el que
  se abre en una planilla para mirar el conjunto.

Las filas de detecciones se escriben en streaming, no se acumulan: son cientos de
miles y guardarlas en memoria antes de volcarlas ya causo un OOM en este proyecto.
"""

from __future__ import annotations

import csv
import json

CAMPOS_DETECCION = [
    "clip_id", "frame", "track_id", "x1", "y1", "x2", "y2",
    "confianza", "confirmado", "reid_sim", "track_id_bytetrack",
]

CAMPOS_CLIP = [
    "clip_id", "location_type", "width", "height", "fps_video",
    "frames_totales", "frames_analizados", "frames_descartados", "pct_analizado",
    "detecciones", "personas_unicas", "max_simultaneas", "conf_media",
    "entradas", "salidas", "cruzaron_linea", "reenganches_reid",
    "cpu_ms_motion", "gpu_ms_person", "cpu_ms_tracking", "cpu_ms_conteo",
    "gpu_ms_reid", "wall_ms", "fps_pipeline",
    "cpu_ms_por_frame", "gpu_ms_por_frame_analizado",
    "costo_usd_cpu", "costo_usd_gpu",
]


class EscritorDetecciones:
    """Escribe una fila por deteccion, en streaming.

    Se usa como `sink` del runner: `procesar_clip(..., sink=escritor.para(clip_id))`.
    """

    def __init__(self, path):
        self.path = path
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=CAMPOS_DETECCION)
        self._w.writeheader()
        self.n = 0

    def para(self, clip_id: str):
        """Devuelve el callback que el runner llama por cada frame analizado."""

        def sink(frame_idx: int, dets) -> None:
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                self._w.writerow({
                    "clip_id": clip_id, "frame": frame_idx,
                    "track_id": d.track_id if d.track_id is not None else "",
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "confianza": round(d.score, 4),
                    "confirmado": int(bool(d.meta.get("confirmado"))),
                    "reid_sim": d.meta.get("reid_sim", ""),
                    "track_id_bytetrack": d.meta.get("track_id_bytetrack", ""),
                })
                self.n += 1

        return sink

    def cerrar(self) -> None:
        self._f.close()


def escribir_resumen(path, stats_por_clip, clips, cfg) -> int:
    """Una fila por clip con conteo, tiempos y costes. Devuelve cuantas escribio."""
    por_id = {c.clip_id: c for c in clips}
    hora = 3_600_000.0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_CLIP)
        w.writeheader()
        for s in sorted(stats_por_clip, key=lambda s: s.clip_id):
            c = por_id.get(s.clip_id)
            cpu_ms = (s.cpu_ms_motion + s.cpu_ms_tracking + s.cpu_ms_conteo)
            gpu_ms = s.gpu_ms_person + s.gpu_ms_reid
            w.writerow({
                "clip_id": s.clip_id,
                "location_type": c.location_type.value if c else "",
                "width": c.width if c else "", "height": c.height if c else "",
                "fps_video": c.fps if c else "",
                "frames_totales": s.total_frames,
                "frames_analizados": s.frames_analizados,
                "frames_descartados": s.frames_descartados,
                "pct_analizado": s.pct_analizado,
                "detecciones": s.n_detecciones,
                "personas_unicas": s.n_personas_unicas,
                "max_simultaneas": s.max_personas_simultaneas,
                "conf_media": s.conf_media,
                "entradas": s.entradas, "salidas": s.salidas,
                "cruzaron_linea": s.contados_por_linea,
                "reenganches_reid": s.n_reenganches,
                "cpu_ms_motion": s.cpu_ms_motion,
                "gpu_ms_person": s.gpu_ms_person,
                "cpu_ms_tracking": s.cpu_ms_tracking,
                "cpu_ms_conteo": s.cpu_ms_conteo,
                "gpu_ms_reid": s.gpu_ms_reid,
                "wall_ms": s.wall_ms, "fps_pipeline": s.fps,
                "cpu_ms_por_frame": round(cpu_ms / s.total_frames, 3) if s.total_frames else 0,
                "gpu_ms_por_frame_analizado": round(gpu_ms / s.frames_analizados, 3)
                if s.frames_analizados else 0,
                # El costo en dolares es 0 mientras las tarifas esten en 0, que es
                # el default. Va igual como columna para que el CSV no cambie de
                # forma cuando se fije una tarifa de verdad.
                "costo_usd_cpu": round(cpu_ms / hora * cfg.cpu_usd_per_hour, 10),
                "costo_usd_gpu": round(gpu_ms / hora * cfg.gpu_usd_per_hour, 10),
            })
    return len(stats_por_clip)


def escribir_metricas(path, eval_stats: dict) -> int:
    """Una fila por umbral de confianza, con precision, recall y F1."""
    campos = ["ambito", "conf", "tp", "fp", "fn", "precision", "recall", "f1"]
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for ambito in ("detector_solo", "sistema_completo"):
            for u, c in sorted(eval_stats[ambito].items(), key=lambda kv: float(kv[0])):
                tp, fp, fn = c["tp"], c["fp"], c["fn"]
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                w.writerow({
                    "ambito": ambito, "conf": float(u), "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                })
                n += 1
    return n


def cargar_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
