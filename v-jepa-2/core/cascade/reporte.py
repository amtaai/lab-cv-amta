"""Genera el reporte del Nivel 1, con la misma forma que froth_gate/results/gate_report.md:
titulo, linea de protocolo, secciones y ## Veredicto al final.
"""

from __future__ import annotations

from core.cascade.catalog.schema import ClipMeta

MIN_CLIPS_COMERCIALES = 60  # objetivo del corpus declarado en el plan de la semana
# Por debajo de esta duracion mediana, el corpus es de segmentos curados y no de
# grabacion continua. Cambia por completo como se lee el % de movimiento.
DURACION_CONTINUA_MIN_S = 60.0
# Un clip por debajo de este % de movimiento cuenta como "escena vacia": es
# metraje de que-no-pasa-nada, el regimen que falta en un corpus curado.
UMBRAL_ESCENA_VACIA_PCT = 10.0


def construir_reporte(stats_por_clip: list, clips: list[ClipMeta],
                      resumen_costo: dict, cfg) -> str:
    """Devuelve el markdown del reporte de Nivel 1."""
    por_id = {c.clip_id: c for c in clips}
    comerciales = [s for s in stats_por_clip if por_id[s.clip_id].es_comercial]
    n_com = len(comerciales)

    lineas: list[str] = []
    lineas.append("# Cascade Nivel 1 (motion detection) — reporte")
    lineas.append("")
    lineas.append(
        f"protocolo: MOG2 history={cfg.motion.history} varThreshold={cfg.motion.var_threshold} "
        f"min_area_frac={cfg.motion.min_area_frac} warmup={cfg.motion.warmup_frames} "
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

    # ---- Estimacion por regimen de actividad -------------------------------
    # El corpus son clips cortos y curados, asi que su tasa global esta inflada.
    # Pero contiene los DOS regimenes grabados por la misma camara fija, y con
    # ellos se estima cualquier ciclo de actividad sin grabar nada nuevo.
    vacios = [s for s in stats_por_clip if s.pct_motion < UMBRAL_ESCENA_VACIA_PCT]
    activos = [s for s in stats_por_clip if s.pct_motion >= UMBRAL_ESCENA_VACIA_PCT]

    def _tasa(grupo):
        k = sum(s.frames_kept for s in grupo)
        d = sum(s.frames_discarded for s in grupo)
        return (k / (k + d)) if (k + d) else 0.0, len(grupo), k + d

    p_v, n_v, f_v = _tasa(vacios)
    p_a, n_a, f_a_frames = _tasa(activos)

    lineas.append("## Estimacion por ciclo de actividad")
    lineas.append("")
    lineas.append(f"- escena vacia : n={n_v} clips, {f_v} frames utiles, p_v = {p_v:.4f}")
    lineas.append(f"- escena activa: n={n_a} clips, {f_a_frames} frames utiles, p_a = {p_a:.4f}")
    lineas.append("")
    lineas.append("| actividad del dia | tasa de paso | descarte | Nivel 2 debe costar > |")
    lineas.append("|---|---|---|---|")
    cpu_frame = (sum(s.cpu_time_ms for s in stats_por_clip) / frames_totales) if frames_totales else 0.0
    for f_act in (0.05, 0.10, 0.15, 0.25, 0.50):
        p = f_act * p_a + (1 - f_act) * p_v
        umbral = cpu_frame / (1 - p) if p < 1 else float("inf")
        lineas.append(f"| {f_act*100:.0f} % | {p:.4f} | {100*(1-p):.1f} % | {umbral:.1f} ms |")
    lineas.append("")

    lineas.append("## Veredicto")
    lineas.append("")
    agg = sum(s.frames_kept for s in stats_por_clip)
    tot = sum(s.frames_kept + s.frames_discarded for s in stats_por_clip)
    pct = 100.0 * agg / tot if tot else 0.0
    lineas.append(
        f"**{pct:.2f} % de los frames del corpus** contienen movimiento relevante "
        f"(n={len(stats_por_clip)} clips, {tot} frames utiles)."
    )
    lineas.append("")
    if n_com < MIN_CLIPS_COMERCIALES:
        n_otros = len(stats_por_clip) - n_com
        lineas.append(
            f"> **ALCANCE.** {n_otros} de los {len(stats_por_clip)} clips estan indexados como "
            "`location_type=other`: NO son interiores comerciales. Esta cifra describe el "
            "corpus disponible, no un comercio, y no debe citarse como tal. Para un local "
            "real se usa la estimacion por ciclo de actividad de la seccion anterior, que "
            "solo depende de p_a, p_v y del ciclo de operacion del local. "
            "Ver `documentation/cascade_semana1.tex` para el detalle de por que no hay "
            "material comercial utilizable."
        )
    return "\n".join(lineas)
