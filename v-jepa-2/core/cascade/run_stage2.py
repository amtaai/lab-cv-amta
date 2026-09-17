"""Corre el cascade completo (Nivel 1 + Nivel 2 + seguimiento) sobre el corpus.

Uso:
  python -m core.cascade.run_stage2
  python -m core.cascade.run_stage2 --clips 20        # subconjunto, para probar
  python -m core.cascade.run_stage2 --rtsp            # ademas, 300 frames del simulador
"""

from __future__ import annotations

import argparse
import json
import time

import cv2

from core.cascade.catalog.index import cargar_indice, verificar_manifest
from core.cascade.config import load_cascade_config
from core.cascade.cost.tracker import CostTracker
from core.cascade.eval.exportar import EscritorDetecciones, escribir_resumen
from core.cascade.reporte2 import construir_reporte
from core.perception.filas import cargar_rois
from core.cascade.stage2_person.runner import (
    construir_detector,
    procesar_clip,
    procesar_stream,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Barrido del Nivel 2 sobre el corpus")
    ap.add_argument("--clips", type=int, default=0, help="usar solo los primeros N clips")
    ap.add_argument("--solo", default="", help="solo los clips cuyo id contenga este texto")
    ap.add_argument("--max-frames", type=int, default=0, help="tope de frames por clip")
    ap.add_argument("--rtsp", action="store_true", help="ademas, medir sobre el stream RTSP")
    ap.add_argument("--rtsp-frames", type=int, default=300)
    ap.add_argument("--sin-filtro", action="store_true",
                    help="saltea el Nivel 1: el detector corre sobre TODOS los "
                         "frames. Es la corrida de control del punto 7")
    ap.add_argument("--sufijo", default="",
                    help="sufijo para los archivos de salida, para no pisar la "
                         "corrida normal cuando se compara")
    args = ap.parse_args()

    cfg = load_cascade_config()
    # Un solo hilo de OpenCV: la medicion de CPU tiene que ser reproducible y
    # comparable contra la del Nivel 1.
    cv2.setNumThreads(cfg.cv_num_threads)

    clips = cargar_indice(cfg.corpus_dir)
    if not clips:
        print("ABORTA: corpus vacio. Ingestar clips con core.cascade.catalog.ingest")
        return 1
    malos = verificar_manifest(cfg.corpus_dir)
    if malos:
        print(f"AVISO: {len(malos)} clips no coinciden con el manifest: {malos}")
    if args.solo:
        clips = [c for c in clips if args.solo in c.clip_id]
        if not clips:
            print(f"ABORTA: ningun clip contiene '{args.solo}'")
            return 1
    if args.clips:
        clips = clips[:args.clips]

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    tracker = CostTracker(cfg.costs_db, cpu_usd_per_hour=cfg.cpu_usd_per_hour,
                          gpu_usd_per_hour=cfg.gpu_usd_per_hour)
    # Un solo modelo para todo el barrido: levantarlo en GPU cuesta segundos y
    # hacerlo por clip domina el tiempo total y ensucia el ms/frame.
    detector = construir_detector(cfg)
    print(f"detector: {cfg.person.backend} device={cfg.person.device} "
          f"imgsz={cfg.person.imgsz} | tracker: {cfg.track.tracker}")
    # Inferencia de descarte antes de medir nada: la primera llamada a CUDA paga
    # la inicializacion y, sin esto, ese costo se le carga al primer clip que
    # tenga movimiento y el C2 del reporte sale inflado.
    if hasattr(detector, "calentar"):
        print(f"calentando la GPU: {detector.calentar():.0f} ms la primera inferencia")

    rois = cargar_rois(cfg.filas.rois_path) if cfg.filas.activo else {}
    if rois:
        print(f"ROIs de zona de espera declaradas: {len(rois)} clips")
    sufijo = args.sufijo or ("_sin_filtro" if args.sin_filtro else "")
    escritor = EscritorDetecciones(cfg.results_dir / f"detecciones{sufijo}.csv")

    stats = []
    t0 = time.time()
    for i, c in enumerate(clips, 1):
        s = procesar_clip(cfg.raw_dir / c.filename, c.clip_id, cfg, tracker,
                          detector=detector, max_frames=args.max_frames,
                          sink=escritor.para(c.clip_id),
                          usar_gate=not args.sin_filtro,
                          roi=rois.get(c.clip_id))
        stats.append(s)
        print(f"{i}/{len(clips)} {c.clip_id}: {s.total_frames} frames | "
              f"al N2 {s.pct_analizado:.0f}% | personas {s.n_personas_unicas} "
              f"({s.n_detecciones} cajas, conf {s.conf_media:.2f}) | "
              f"N2 {s.ms_por_frame_analizado:.0f} ms gpu/frame | "
              f"{s.fps:.1f} fps", flush=True)

    if args.rtsp:
        print(f"midiendo {args.rtsp_frames} frames de {cfg.rtsp_url} ...")
        s = procesar_stream(cfg.rtsp_url, cfg, tracker, detector=detector,
                            max_frames=args.rtsp_frames)
        print(f"rtsp: {s.total_frames} frames | al N2 {s.pct_analizado:.0f}% | "
              f"personas {s.n_personas_unicas} | {s.meta.get('error', '')}")

    escritor.cerrar()
    resumen = tracker.resumen_por_stage()
    tracker.close()

    payload = {
        "protocolo": {
            "backend": cfg.person.backend,
            "pesos": cfg.person.pesos or "(default del backend)",
            "prompts": list(cfg.person.prompts),
            "imgsz": cfg.person.imgsz,
            "conf": cfg.person.conf,
            "iou": cfg.person.iou,
            "device": cfg.person.device,
            "tracker": cfg.track.tracker,
            "min_hits": cfg.track.min_hits,
            "max_age": cfg.track.max_age,
            "linea_conteo": list(cfg.conteo.linea),
            "cv_num_threads": cfg.cv_num_threads,
            "min_area_frac": cfg.motion.min_area_frac,
            "gate_activo": not args.sin_filtro,
            "reid_activo": cfg.reid.activo,
            "cpu_usd_per_hour": cfg.cpu_usd_per_hour,
            "gpu_usd_per_hour": cfg.gpu_usd_per_hour,
        },
        "n_clips": len(stats),
        "t_total_s": round(time.time() - t0, 3),
        "por_clip": [
            {**s.__dict__,
             "pct_analizado": s.pct_analizado,
             "pct_frames_con_persona": s.pct_frames_con_persona,
             "ms_por_frame_analizado": s.ms_por_frame_analizado,
             "fps_pipeline": s.fps}
            for s in stats
        ],
        "costo_por_stage": resumen,
    }
    (cfg.results_dir / f"stage2_person_stats{sufijo}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    escribir_resumen(cfg.results_dir / f"resumen_clips{sufijo}.csv", stats, clips, cfg)
    print(f"\nCSV: detecciones{sufijo}.csv ({escritor.n} filas) | "
          f"resumen_clips{sufijo}.csv ({len(stats)} filas)")

    reporte = construir_reporte(stats, clips, resumen, cfg)
    (cfg.results_dir / f"stage2_person_report{sufijo}.md").write_text(
        reporte, encoding="utf-8")
    # Se imprime todo menos la tabla por clip, que con 182 filas tapa la consola.
    cabeza, _, resto = reporte.partition("## Por clip")
    print()
    print(cabeza + "## Deteccion" + resto.partition("## Deteccion")[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
