"""Compara el pipeline con y sin el filtro de movimiento (punto 7).

Hasta ahora el "sin cascada" del reporte estaba EXTRAPOLADO: frames totales por el
coste por frame. Esto lo mide de verdad, corriendo el detector sobre todos los
frames, y compara frames procesados, tiempo de CPU y GPU, FPS y coste.

Requiere las dos corridas hechas antes:
  python -m core.cascade.run_stage2 --solo retail_ --sufijo _con_filtro
  python -m core.cascade.run_stage2 --solo retail_ --sin-filtro

Uso:
  python -m core.cascade.eval.comparar_gate
"""

from __future__ import annotations

import json

from core.cascade.config import load_cascade_config

HORA_MS = 3_600_000.0


def _totales(d: dict) -> dict:
    c = d["costo_por_stage"]
    mog2 = c["motion_detection"]["cpu_total_ms"]
    p = d["por_clip"]
    cpu = sum(v["cpu_total_ms"] for v in c.values())
    gpu = sum(v.get("gpu_total_ms", 0.0) for v in c.values())
    wall = sum(x["wall_ms"] for x in p)
    tot = sum(x["total_frames"] for x in p)
    return {
        "frames_totales": tot,
        "frames_al_detector": sum(x["frames_analizados"] for x in p),
        "cpu_ms": cpu, "gpu_ms": gpu, "wall_ms": wall,
        "fps": tot / (wall / 1000.0) if wall else 0.0,
        "personas": sum(x["n_personas_unicas"] for x in p),
        "detecciones": sum(x["n_detecciones"] for x in p),
        "cpu_mog2": mog2,
        # CPU sin contar MOG2. Hace falta para una comparacion justa: en la corrida
        # "sin filtro" MOG2 igual se ejecuta (solo se ignora su veredicto), y un
        # sistema que de verdad no tuviera cascada no lo correria en absoluto.
        "cpu_sin_mog2": cpu - mog2,
    }


def main() -> int:
    cfg = load_cascade_config()
    a = cfg.results_dir / "stage2_person_stats_con_filtro.json"
    b = cfg.results_dir / "stage2_person_stats_sin_filtro.json"
    for p in (a, b):
        if not p.exists():
            print(f"ABORTA: falta {p.name}. Ver el docstring de este modulo.")
            return 1

    con = _totales(json.loads(a.read_text(encoding="utf-8")))
    sin = _totales(json.loads(b.read_text(encoding="utf-8")))

    L = ["# Coste con y sin el filtro de movimiento — reporte", ""]
    L.append("Las dos corridas son sobre los mismos clips y con el mismo detector. "
             "La unica diferencia es si el Nivel 1 decide que frames llegan al "
             "detector o si le llegan todos.")
    L.append("")
    L.append("| | con cascada | sin cascada | diferencia |")
    L.append("|---|---|---|---|")
    filas = [
        ("frames del video", "frames_totales", "{:,.0f}"),
        ("frames que vio el detector", "frames_al_detector", "{:,.0f}"),
        ("detecciones producidas", "detecciones", "{:,.0f}"),
        ("CPU (ms)", "cpu_ms", "{:,.0f}"),
        ("GPU (ms)", "gpu_ms", "{:,.0f}"),
        ("reloj de pared (s)", "wall_ms", None),
        ("FPS del pipeline", "fps", "{:.1f}"),
    ]
    for nombre, clave, fmt in filas:
        va, vb = con[clave], sin[clave]
        if clave == "wall_ms":
            va, vb, fmt = va / 1000, vb / 1000, "{:,.1f}"
        dif = (100 * (va - vb) / vb) if vb else 0.0
        L.append(f"| {nombre} | {fmt.format(va)} | {fmt.format(vb)} | {dif:+.1f} % |")
    L.append("")

    ahorro_gpu = 100 * (1 - con["gpu_ms"] / sin["gpu_ms"]) if sin["gpu_ms"] else 0.0
    L.append(f"**El filtro ahorra {ahorro_gpu:.1f} % de GPU** y el pipeline corre "
             f"{con['fps'] / sin['fps'] if sin['fps'] else 0:.2f}x mas rapido "
             f"({con['fps']:.1f} contra {sin['fps']:.1f} FPS).")
    L.append("")
    # El intercambio se explica en milisegundos por frame, no en porcentaje: el
    # denominador "CPU sin MOG2" es casi cero (solo tracking y conteo) y cualquier
    # porcentaje contra el sale en miles, que no le dice nada a nadie.
    fr = con["frames_totales"]
    mog2_por_frame = con["cpu_mog2"] / fr if fr else 0.0
    gpu_ahorrada = (sin["gpu_ms"] - con["gpu_ms"]) / fr if fr else 0.0
    L.append("### El intercambio, en milisegundos por frame")
    L.append("")
    L.append(f"- el gate **cuesta {mog2_por_frame:.1f} ms de CPU** en cada frame, "
             "porque MOG2 corre sobre todos")
    L.append(f"- y **ahorra {gpu_ahorrada:.1f} ms de GPU** por frame")
    L.append("")
    L.append("No es un buen trato en tiempo total —se gasta mas CPU de la que se "
             "ahorra en GPU— y aun asi conviene, porque **los dos recursos no valen "
             "lo mismo**: la GPU es la que limita cuantas camaras entran por maquina "
             "y la CPU suele sobrar. El intercambio es deliberado.")
    L.append("")
    L.append("A 720p MOG2 sale caro: "
             f"{mog2_por_frame:.1f} ms/frame contra los 8,9 ms medidos sobre el corpus "
             "completo, que es casi todo 480p. A mayor resolucion el gate se encarece "
             "y habria que medir de nuevo si sigue conviniendo.")
    L.append("")
    L.append("### Por que la CPU total tambien baja")
    L.append("")
    ahorro_cpu_real = (sin["cpu_sin_mog2"] - con["cpu_sin_mog2"])
    L.append(f"La tabla muestra menos CPU con cascada, lo que confunde: MOG2 corre "
             f"sobre todos los frames en las DOS corridas. Lo que de verdad se ahorra "
             f"son {ahorro_cpu_real:,.0f} ms de registro de tracks y conteo, que "
             f"tampoco corren sobre los frames descartados. El resto de la diferencia "
             f"es ruido de medicion entre corridas "
             f"({abs(sin['cpu_mog2'] - con['cpu_mog2']):,.0f} ms en MOG2, "
             f"{100 * abs(sin['cpu_mog2'] - con['cpu_mog2']) / con['cpu_mog2']:.1f} %).")
    L.append("")
    L.append("> La CPU se mide con `time.thread_time_ns()`. Con `process_time_ns()`, "
             "que es lo que usaba la Semana 1, el mismo trabajo de MOG2 daba 133.268 "
             "y 153.621 ms segun cuanto trabajara el detector en paralelo: process_time "
             "cuenta los hilos que levanta torch y los sumaba al Nivel 1.")
    L.append("")
    L.append("## Coste en dolares")
    L.append("")
    if cfg.cpu_usd_per_hour == 0.0 and cfg.gpu_usd_per_hour == 0.0:
        L.append("> Las dos tarifas estan en 0.0, que es el default a proposito, asi "
                 "que **el coste en dolares es 0 por construccion**. Los milisegundos "
                 "si son reales. Con `AMTA_CPU_USD_PER_HOUR` y `AMTA_GPU_USD_PER_HOUR` "
                 "fijadas a una tarifa verificada, esta tabla se llena sola:")
        L.append("")
    L.append("| | con cascada | sin cascada |")
    L.append("|---|---|---|")
    L.append(f"| USD de CPU | {con['cpu_ms'] / HORA_MS * cfg.cpu_usd_per_hour:.8f} | "
             f"{sin['cpu_ms'] / HORA_MS * cfg.cpu_usd_per_hour:.8f} |")
    L.append(f"| USD de GPU | {con['gpu_ms'] / HORA_MS * cfg.gpu_usd_per_hour:.8f} | "
             f"{sin['gpu_ms'] / HORA_MS * cfg.gpu_usd_per_hour:.8f} |")
    L.append("")

    L.append("## Que se pierde")
    L.append("")
    L.append(f"- personas contadas: **{con['personas']}** con cascada contra "
             f"**{sin['personas']}** sin ella")
    L.append(f"- detecciones: {con['detecciones']:,} contra {sin['detecciones']:,}")
    L.append("")
    L.append("El ahorro no es gratis: los frames que el Nivel 1 descarta pueden tener "
             "gente. Cuanto cuesta eso en recall esta medido aparte, en "
             "`eval_report.md` y en `errores_report.md`.")

    (cfg.results_dir / "coste_gate_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
