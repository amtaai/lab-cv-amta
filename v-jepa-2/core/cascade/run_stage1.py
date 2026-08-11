"""Corre el Nivel 1 sobre todo el corpus y emite stats + reporte.

Uso:
  python -m core.cascade.run_stage1
  python -m core.cascade.run_stage1 --rtsp     # ademas, 300 frames del simulador
"""

from __future__ import annotations

import argparse
import json
import time

import cv2

from core.cascade.catalog.index import cargar_indice, verificar_manifest
from core.cascade.config import load_cascade_config
from core.cascade.cost.tracker import CostTracker
from core.cascade.reporte import construir_reporte
from core.cascade.stage1_motion.runner import procesar_clip, procesar_stream


def main() -> int:
    ap = argparse.ArgumentParser(description="Barrido de Nivel 1 sobre el corpus")
    ap.add_argument("--rtsp", action="store_true", help="ademas, medir sobre el stream RTSP")
    ap.add_argument("--rtsp-frames", type=int, default=300)
    args = ap.parse_args()

    cfg = load_cascade_config()
    # Un solo hilo de OpenCV: la medicion de CPU time tiene que ser reproducible.
    cv2.setNumThreads(cfg.cv_num_threads)

    clips = cargar_indice(cfg.corpus_dir)
    if not clips:
        print("ABORTA: corpus vacio. Ingestar clips con core.cascade.catalog.ingest")
        return 1

    malos = verificar_manifest(cfg.corpus_dir)
    if malos:
        print(f"AVISO: {len(malos)} clips no coinciden con el manifest: {malos}")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    tracker = CostTracker(cfg.costs_db, cpu_usd_per_hour=cfg.cpu_usd_per_hour)

    stats = []
    t0 = time.time()
    for i, c in enumerate(clips, 1):
        s = procesar_clip(cfg.raw_dir / c.filename, c.clip_id, cfg, tracker)
        stats.append(s)
        print(f"{i}/{len(clips)} {c.clip_id}: {s.total_frames} frames | "
              f"movimiento {s.pct_motion:.1f}% | flicker {s.flicker_suspect_frames} | "
              f"{s.cpu_time_ms:.0f} ms cpu", flush=True)

    if args.rtsp:
        print(f"midiendo {args.rtsp_frames} frames de {cfg.rtsp_url} ...")
        s = procesar_stream(cfg.rtsp_url, cfg, tracker, max_frames=args.rtsp_frames)
        print(f"rtsp: {s.total_frames} frames | movimiento {s.pct_motion:.1f}% | "
              f"{s.cpu_time_ms:.0f} ms cpu | {s.meta.get('error','')}")

    resumen = tracker.resumen_por_stage()
    tracker.close()

    payload = {
        "protocolo": {
            "history": cfg.motion.history,
            "var_threshold": cfg.motion.var_threshold,
            "min_area_px": cfg.motion.min_area_px,
            "warmup_frames": cfg.motion.warmup_frames,
            "flicker_fg_ratio": cfg.motion.flicker_fg_ratio,
            "cv_num_threads": cfg.cv_num_threads,
            "cpu_usd_per_hour": cfg.cpu_usd_per_hour,
        },
        "n_clips": len(stats),
        "t_total_s": round(time.time() - t0, 3),
        "por_clip": [s.__dict__ for s in stats],
        "costo_por_stage": resumen,
    }
    (cfg.results_dir / "stage1_motion_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    reporte = construir_reporte(stats, clips, resumen, cfg)
    (cfg.results_dir / "stage1_motion_report.md").write_text(reporte, encoding="utf-8")
    print()
    print(reporte)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
