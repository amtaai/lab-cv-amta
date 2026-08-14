"""Genera el reporte del Nivel 1, con la misma forma que froth_gate/results/gate_report.md:
titulo, linea de protocolo, secciones y ## Veredicto al final.
"""

from __future__ import annotations

from core.cascade.catalog.schema import ClipMeta, LocationType

MIN_CLIPS_COMERCIALES = 60  # objetivo del corpus declarado en el plan de la semana


def _es_comercial(c: ClipMeta) -> bool:
    return c.location_type is not LocationType.OTHER


def construir_reporte(stats_por_clip: list, clips: list[ClipMeta],
                      resumen_costo: dict, cfg) -> str:
    """Devuelve el markdown del reporte de Nivel 1."""
    por_id = {c.clip_id: c for c in clips}
    comerciales = [s for s in stats_por_clip if _es_comercial(por_id[s.clip_id])]
    n_com = len(comerciales)

    lineas: list[str] = []
    lineas.append("# Cascade Nivel 1 (motion detection) — reporte")
    lineas.append("")
    lineas.append(
        f"protocolo: MOG2 history={cfg.motion.history} varThreshold={cfg.motion.var_threshold} "
        f"min_area_px={cfg.motion.min_area_px} warmup={cfg.motion.warmup_frames} "
        f"flicker_fg_ratio={cfg.motion.flicker_fg_ratio} | cv_threads={cfg.cv_num_threads}"
    )
    lineas.append("")

    lineas.append("## Por clip")
    lineas.append("")
    lineas.append("| clip_id | tipo | frames | % movimiento | % descartado | flicker | cpu_ms |")
    lineas.append("|---|---|---|---|---|---|---|")
    for s in sorted(stats_por_clip, key=lambda s: s.clip_id):
        tipo = por_id[s.clip_id].location_type.value
        lineas.append(
            f"| {s.clip_id} | {tipo} | {s.total_frames} | {s.pct_motion:.2f} | "
            f"{s.pct_discarded:.2f} | {s.flicker_suspect_frames} | {s.cpu_time_ms:.1f} |"
        )
    lineas.append("")

    lineas.append("## Costo (Nivel 1)")
    lineas.append("")
    frames_totales = sum(s.total_frames for s in stats_por_clip)
    for stage, r in resumen_costo.items():
        lineas.append(
            f"- **{stage}**: {r['n_eventos']} eventos | cpu medio {r['cpu_medio_ms']:.2f} ms/evento "
            f"| cpu total {r['cpu_total_ms']:.1f} ms | costo USD {r['cost_usd_total']:.8f}"
        )
    # El evento es por clip, asi que ms/evento no se puede comparar entre clips de
    # distinto largo. La normalizacion util —y la que alimenta la proyeccion de
    # costo mensual por camara— es por frame.
    # Se calcula sobre los clips de la tabla de arriba, NO sobre resumen_costo:
    # ese incluye tambien el evento del stream RTSP, cuyos frames no estan en
    # frames_totales, y el ms/frame saldria inflado.
    if frames_totales:
        cpu_clips = sum(s.cpu_time_ms for s in stats_por_clip)
        lineas.append(
            f"- **por frame**: {cpu_clips / frames_totales:.2f} ms cpu/frame "
            f"({frames_totales} frames de archivo, sin contar el stream RTSP)"
        )
    if cfg.cpu_usd_per_hour == 0.0:
        lineas.append("")
        lineas.append(
            "> `AMTA_CPU_USD_PER_HOUR` esta en 0.0, asi que **cost_usd es 0 por construccion**. "
            "Los tiempos de CPU si son reales. Fijar una tarifa cloud verificada antes de "
            "citar cualquier costo en dolares."
        )
    lineas.append("")

    # La guarda de flicker no distingue "se movio la luz" de "se movio la camara":
    # las dos encienden el frame entero. En clips de camara en mano el contador de
    # flicker mide movimiento de camara, no parpadeo. Hay que decirlo o el numero
    # se lee como si fuera iluminacion.
    if any(s.flicker_suspect_frames for s in stats_por_clip):
        lineas.append(
            "> La guarda de flicker dispara con cualquier cambio global del frame, y no "
            "puede separar un cambio de luz de un movimiento de camara. En clips de camara "
            "en mano el contador de flicker mide lo segundo. En CCTV fijo —el caso real— la "
            "camara no se mueve, asi que ahi si aisla iluminacion."
        )
        lineas.append("")

    lineas.append("## Veredicto")
    lineas.append("")
    if n_com < MIN_CLIPS_COMERCIALES:
        n_otros = len(stats_por_clip) - n_com
        detalle = (
            f" ({n_otros} clips mas estan indexados como `location_type=other` y no cuentan:"
            " no son interiores comerciales)." if n_otros else "."
        )
        lineas.append(
            f"**PENDIENTE — corpus insuficiente.** Hay {n_com} clips de interior comercial "
            f"sobre un objetivo de {MIN_CLIPS_COMERCIALES}{detalle} El % de frames con "
            "movimiento relevante NO se publica hasta llegar al objetivo: con menos clips "
            "el numero no generaliza. La instrumentacion de costo SI esta validada."
        )
    else:
        agg = sum(s.frames_kept for s in comerciales)
        tot = sum(s.frames_kept + s.frames_discarded for s in comerciales)
        pct = 100.0 * agg / tot if tot else 0.0
        lineas.append(
            f"**{pct:.2f} % de los frames** de un interior comercial contienen movimiento "
            f"relevante (n={n_com} clips, {tot} frames utiles)."
        )
    return "\n".join(lineas)
