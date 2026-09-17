"""Reporte del Nivel 2 (deteccion de personas + seguimiento).

Misma forma que el del Nivel 1: titulo, linea de protocolo, secciones y
## Veredicto al final, para que ningun numero se pueda separar de como se produjo.

Ojo con las unidades: el Nivel 1 se mide en ms de CPU y el Nivel 2 en ms de GPU.
No se suman. Son recursos distintos, con precios distintos, y un unico "ms/frame"
que los mezclara escondería cual de los dos es el que hay que pagar.
"""

from __future__ import annotations

from core.cascade.catalog.schema import ClipMeta

MIN_CLIPS_COMERCIALES = 60  # el mismo objetivo declarado para el Nivel 1
# Debajo de este % de frames analizados, el clip es escena vacia: el Nivel 1 casi
# no dejo pasar nada. Es el mismo corte que usa el reporte del Nivel 1.
UMBRAL_ESCENA_VACIA_PCT = 10.0
FPS_CAMARA = 25.0  # para expresar el costo como fraccion de un recurso por camara


def construir_reporte(stats_por_clip: list, clips: list[ClipMeta],
                      resumen_costo: dict, cfg) -> str:
    """Devuelve el markdown del reporte del Nivel 2."""
    por_id = {c.clip_id: c for c in clips}
    n_com = sum(1 for s in stats_por_clip if por_id[s.clip_id].es_comercial)

    frames_totales = sum(s.total_frames for s in stats_por_clip)
    frames_utiles = sum(s.frames_utiles for s in stats_por_clip)
    frames_analizados = sum(s.frames_analizados for s in stats_por_clip)
    frames_persona = sum(s.frames_con_persona for s in stats_por_clip)
    n_det = sum(s.n_detecciones for s in stats_por_clip)
    n_personas = sum(s.n_personas_unicas for s in stats_por_clip)

    cpu_mov = sum(s.cpu_ms_motion for s in stats_por_clip)
    gpu_per = sum(s.gpu_ms_person for s in stats_por_clip)
    cpu_reg = sum(s.cpu_ms_tracking for s in stats_por_clip)
    cpu_cnt = sum(getattr(s, "cpu_ms_conteo", 0.0) for s in stats_por_clip)
    gpu_rid = sum(getattr(s, "gpu_ms_reid", 0.0) for s in stats_por_clip)
    reeng = sum(getattr(s, "n_reenganches", 0) for s in stats_por_clip)
    entradas = sum(s.entradas for s in stats_por_clip)
    salidas = sum(s.salidas for s in stats_por_clip)
    cruzaron = sum(s.contados_por_linea for s in stats_por_clip)

    # C1 se paga por TODO frame (CPU); C2 solo por los que pasaron (GPU).
    c1 = cpu_mov / frames_totales if frames_totales else 0.0
    c2 = gpu_per / frames_analizados if frames_analizados else 0.0
    p = frames_analizados / frames_utiles if frames_utiles else 0.0

    L: list[str] = []
    L.append("# Cascade Nivel 2 (deteccion de personas + seguimiento) — reporte")
    L.append("")
    L.append(
        f"protocolo: {cfg.person.backend} pesos={cfg.person.pesos or '(default)'} "
        f"imgsz={cfg.person.imgsz} conf={cfg.person.conf} iou={cfg.person.iou} "
        f"device={cfg.person.device} prompts={list(cfg.person.prompts)} | "
        f"seguimiento {cfg.track.tracker} min_hits={cfg.track.min_hits} "
        f"max_age={cfg.track.max_age}"
    )
    L.append("")
    L.append(f"conteo: linea={list(cfg.conteo.linea)} (normalizada)")
    L.append("")
    if cfg.reid.activo:
        from core.perception.reid import ReIdConfig
        r = ReIdConfig()
        L.append(f"Nivel 3 (ReID): {r.modelo} umbral={r.umbral} "
                 f"ventana={r.ventana_s}s mascara={r.usar_mascara}")
    else:
        L.append("Nivel 3 (ReID): APAGADO")
    L.append("")
    L.append(
        f"gate Nivel 1: MOG2 min_area_frac={cfg.motion.min_area_frac} "
        f"warmup={cfg.motion.warmup_frames} flicker_fg_ratio={cfg.motion.flicker_fg_ratio} "
        f"| cv_threads={cfg.cv_num_threads}"
    )
    L.append("")
    L.append("> El detector y el tracker NO son de este modulo: son "
             "`core/perception/`, los mismos que corren los notebooks de "
             "`yolo_seg/` y `yolo_world/`. El cascade los cablea al filtro de "
             "movimiento y les mide el costo.")
    L.append("")

    L.append("## Por clip")
    L.append("")
    L.append("| clip_id | frames | % al Nivel 2 | personas | entra | sale | "
             "conf media | ms gpu/frame |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in sorted(stats_por_clip, key=lambda s: s.clip_id):
        L.append(
            f"| {s.clip_id} | {s.total_frames} | {s.pct_analizado:.2f} | "
            f"{s.n_personas_unicas} | {s.entradas} | {s.salidas} | "
            f"{s.conf_media:.3f} | {s.ms_por_frame_analizado:.1f} |"
        )
    L.append("")

    L.append("## Deteccion")
    L.append("")
    pct_persona = 100.0 * frames_persona / frames_analizados if frames_analizados else 0.0
    conf_global = (sum(s.conf_media * s.n_detecciones for s in stats_por_clip) / n_det
                   if n_det else 0.0)
    L.append(f"- frames analizados por el Nivel 2: **{frames_analizados}** de "
             f"{frames_utiles} utiles ({100*p:.2f} %)")
    L.append(f"- frames con al menos una persona: **{frames_persona}** "
             f"({pct_persona:.2f} % de los analizados)")
    L.append(f"- cajas totales: **{n_det}** | confianza media **{conf_global:.3f}**")
    L.append("")

    L.append("## Seguimiento")
    L.append("")
    L.append(f"- personas unicas (tracks confirmados): **{n_personas}** sumando "
             f"los {len(stats_por_clip)} clips")
    L.append("  - El tracker se reinicia en cada clip, que es lo correcto: son "
             "grabaciones distintas. Asi que este total son personas-por-clip, no "
             "humanos distintos; el mismo actor reaparece en muchos clips.")
    if n_det and n_personas:
        L.append(f"- cajas por persona: **{n_det / n_personas:.1f}** — sin seguimiento "
                 f"esas {n_det} cajas se contarian como {n_det} personas")
    L.append(f"- la asociacion la hace **{cfg.track.tracker}** adentro de ultralytics; "
             f"un track cuenta como persona a partir de {cfg.track.min_hits} detecciones")
    if reeng:
        L.append(f"- el Nivel 3 recupero **{reeng} identidades** que la oclusion habia "
                 "partido. Sin el, cada vez que alguien se tapa detras de una gondola "
                 "y reaparece cuenta como una persona nueva.")
    L.append("")
    L.append("> **Cuanto vale este numero.** El conteo no esta validado contra "
             "etiquetas: no hay ground truth de cuantas personas hay en cada clip "
             "del corpus. Los ms/frame de este reporte son mediciones; el conteo de "
             "personas es una salida del sistema sin verificar. Validarlo pide "
             "etiquetar a mano una muestra de clips.")
    L.append("")

    L.append("## Conteo de personas")
    L.append("")
    L.append("Las tres reglas, para que el numero se pueda auditar:")
    L.append("")
    L.append("- **entra**: el centroide del track cruza la linea de conteo hacia la "
             "derecha (o hacia abajo si la linea fuera horizontal)")
    L.append("- **sale**: el mismo cruce en sentido contrario")
    L.append("- **se contabiliza**: una sola vez por `track_id`, y solo si el registro "
             f"lo confirmo con {cfg.track.min_hits} detecciones. Ir y venir sobre la "
             "linea no infla el numero")
    L.append("")
    L.append(f"- entradas: **{entradas}** | salidas: **{salidas}**")
    L.append(f"- personas distintas que cruzaron la linea: **{cruzaron}**")
    L.append(f"- personas distintas vistas en el cuadro (crucen o no): **{n_personas}**")
    L.append("")
    L.append("Los dos ultimos numeros miden cosas distintas y conviene no confundirlos: "
             "el primero es trafico por un punto y el segundo es presencia. La "
             "diferencia entre ambos es gente que se movio en el cuadro sin llegar a "
             "cruzar la linea.")
    L.append("")

    L.append("## Costo por etapa")
    L.append("")
    for stage, r in resumen_costo.items():
        L.append(
            f"- **{stage}**: {r['n_eventos']} eventos | cpu {r['cpu_total_ms']:.1f} ms "
            f"| gpu {r.get('gpu_total_ms', 0.0):.1f} ms "
            f"| costo USD {r['cost_usd_total']:.8f}"
        )
    L.append("")
    L.append("| etapa | recurso | corre sobre | ms totales | ms/frame |")
    L.append("|---|---|---|---|---|")
    L.append(f"| Nivel 1 · MOG2 | CPU | los {frames_totales} frames | "
             f"{cpu_mov:.0f} | {c1:.2f} |")
    L.append(f"| Nivel 2 · {cfg.person.backend} | GPU | los {frames_analizados} con "
             f"movimiento | {gpu_per:.0f} | "
             f"{gpu_per/frames_analizados if frames_analizados else 0:.2f} |")
    L.append(f"| registro de tracks | CPU | los {frames_analizados} con movimiento | "
             f"{cpu_reg:.0f} | {cpu_reg/frames_analizados if frames_analizados else 0:.3f} |")
    L.append(f"| conteo por linea | CPU | los {frames_analizados} con movimiento | "
             f"{cpu_cnt:.0f} | {cpu_cnt/frames_analizados if frames_analizados else 0:.3f} |")
    if gpu_rid:
        L.append(f"| Nivel 3 · ReID | GPU | solo cuando aparece un ID nuevo | "
                 f"{gpu_rid:.0f} | "
                 f"{gpu_rid/frames_analizados if frames_analizados else 0:.3f} |")
    L.append("")
    if cfg.gpu_usd_per_hour == 0.0:
        L.append("> `AMTA_GPU_USD_PER_HOUR` esta en 0.0, asi que **el costo en dolares "
                 "del Nivel 2 es 0 por construccion**. Los milisegundos de GPU si son "
                 "reales. Fijar una tarifa verificada antes de citar dolares.")
        L.append("")

    # ---- Lo que la cascada ahorra, ahora con C2 medido ----------------------
    L.append("## Economia de la cascada")
    L.append("")
    sin_cascada = frames_totales * c2
    con_cascada = frames_analizados * c2
    ahorro = 100.0 * (1 - con_cascada / sin_cascada) if sin_cascada else 0.0
    L.append(f"- C1 (Nivel 1, CPU) = **{c1:.2f} ms/frame**, sobre el 100 % de los frames")
    L.append(f"- C2 (Nivel 2, GPU) = **{c2:.2f} ms/frame**, solo sobre los que pasan")
    L.append(f"- tasa de paso medida p = **{p:.4f}**")
    L.append("")
    L.append(f"- GPU si el detector corriera sobre todo: {sin_cascada/1000:.0f} s")
    L.append(f"- GPU en cascada: {con_cascada/1000:.0f} s")
    L.append(f"- **ahorro de GPU: {ahorro:.1f} %**")
    L.append("")
    L.append("Con las etapas en recursos distintos, la cascada ya no es un "
             "intercambio de ms contra ms: el Nivel 1 gasta CPU, que sobra, para no "
             "gastar GPU, que es el recurso caro y el que limita cuantas camaras "
             "entran por maquina. El ahorro de GPU es directamente proporcional a "
             "los frames que el Nivel 1 descarta.")
    L.append("")

    # ---- Proyeccion por ciclo de actividad ---------------------------------
    # Igual que en el Nivel 1: el corpus son clips curados donde pasa algo, asi
    # que su p esta inflada respecto de una camara que mira una sala vacia.
    vacios = [s for s in stats_por_clip if s.pct_analizado < UMBRAL_ESCENA_VACIA_PCT]
    activos = [s for s in stats_por_clip if s.pct_analizado >= UMBRAL_ESCENA_VACIA_PCT]

    def _tasa(grupo):
        a = sum(s.frames_analizados for s in grupo)
        u = sum(s.frames_utiles for s in grupo)
        return (a / u if u else 0.0), len(grupo)

    p_v, n_v = _tasa(vacios)
    p_a, n_a = _tasa(activos)

    # ---- comercial vs no comercial ----------------------------------------
    # Es la pregunta que ordena todo el proyecto: los numeros de la oficina, que
    # es de donde sale casi todo el corpus, se transfieren a un local real?
    com = [s for s in stats_por_clip if por_id[s.clip_id].es_comercial]
    otros = [s for s in stats_por_clip if not por_id[s.clip_id].es_comercial]
    if com and otros:
        L.append("## Interior comercial vs el resto del corpus")
        L.append("")
        L.append("| grupo | clips | frames | tasa de paso | ms gpu/frame | "
                 "personas/clip | cajas por persona |")
        L.append("|---|---|---|---|---|---|---|")
        for nombre, grupo in (("interior comercial", com), ("resto (oficina)", otros)):
            ft = sum(x.total_frames for x in grupo)
            fu = sum(x.frames_utiles for x in grupo)
            fa = sum(x.frames_analizados for x in grupo)
            gp = sum(x.gpu_ms_person for x in grupo)
            nd = sum(x.n_detecciones for x in grupo)
            npe = sum(x.n_personas_unicas for x in grupo)
            L.append(f"| {nombre} | {len(grupo)} | {ft} | {fa/fu if fu else 0:.4f} | "
                     f"{gp/fa if fa else 0:.2f} | {npe/len(grupo):.1f} | "
                     f"{nd/npe if npe else 0:.1f} |")
        L.append("")
        L.append("Los ms/frame de GPU dependen del hardware y de la resolucion, no del "
                 "contenido, asi que ahi la comparacion es directa. La tasa de paso si "
                 "depende de la escena, y es el numero que nunca se pudo transferir "
                 "desde la oficina.")
        L.append("")
        L.append(f"> Con {len(com)} clips comerciales esto es un sondeo, no una "
                 "medicion: sirve para ver si los ordenes de magnitud se sostienen, "
                 "no para reemplazar el barrido sobre un corpus comercial de verdad.")
        L.append("")

    L.append("## Cuantas camaras entran por GPU")
    L.append("")
    L.append(f"- escena vacia : n={n_v} clips, p_v = {p_v:.4f}")
    L.append(f"- escena activa: n={n_a} clips, p_a = {p_a:.4f}")
    L.append("")
    L.append(f"| actividad del dia | tasa de paso | ms gpu/frame | ahorro de GPU | "
             f"camaras por GPU a {FPS_CAMARA:.0f} fps |")
    L.append("|---|---|---|---|---|")
    for f_act in (0.05, 0.10, 0.15, 0.25, 0.50):
        pp = f_act * p_a + (1 - f_act) * p_v
        por_frame = pp * c2
        ah = 100.0 * (1 - pp) if c2 else 0.0
        # Una GPU tiene 1000 ms de trabajo por segundo; cada camara consume
        # fps * ms/frame de esos.
        usado = por_frame * FPS_CAMARA
        camaras = 1000.0 / usado if usado > 0 else float("inf")
        L.append(f"| {f_act*100:.0f} % | {pp:.4f} | {por_frame:.2f} | {ah:.1f} % | "
                 f"{camaras:.1f} |")
    L.append("")
    sin_c = c2 * FPS_CAMARA
    L.append(f"Sin cascada entran **{1000.0/sin_c:.1f} camaras** por GPU, sin importar "
             f"si hay alguien o no. Esa es la fila contra la que hay que comparar.")
    L.append("")

    L.append("## Veredicto")
    L.append("")
    L.append(
        f"El Nivel 2 corre sobre el **{100*p:.2f} %** de los frames (el resto lo "
        f"descarta el Nivel 1) y encuentra **{n_personas} personas unicas** en "
        f"{n_det} cajas, con confianza media {conf_global:.3f}. "
        f"La cascada ahorra **{ahorro:.1f} %** de GPU contra correr el detector siempre. "
        f"Cruzaron la linea de conteo **{cruzaron}** personas ({entradas} entradas, "
        f"{salidas} salidas)."
    )
    L.append("")
    if n_com < MIN_CLIPS_COMERCIALES:
        n_otros = len(stats_por_clip) - n_com
        L.append(
            f"> **ALCANCE.** {n_otros} de los {len(stats_por_clip)} clips estan indexados "
            "como `location_type=other`: NO son interiores comerciales. Las cifras "
            "describen el corpus disponible, no un comercio. Los ms/frame de cada etapa "
            "si son transferibles —dependen del hardware y de la resolucion, no del "
            "contenido—; lo que no transfiere es la tasa de paso p, y para eso esta la "
            "tabla por ciclo de actividad."
        )
    return "\n".join(L)
