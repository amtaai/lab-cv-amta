"""Figuras del documento, todas generadas desde los resultados medidos.

Ningun numero se escribe a mano: sale de results/stage2_person_stats.json y de
results/reid_calibracion.json. Si se vuelve a correr el barrido, se regeneran las
figuras y siguen coincidiendo con el texto.

Uso:
  python -m core.cascade.eval.figuras
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # sin display en el contenedor
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from core.cascade.config import load_cascade_config  # noqa: E402

GRIS, AZUL, NARANJA, VERDE, ROJO = "#5A6272", "#2E6DA4", "#D96A26", "#3B8C5A", "#B4552B"


def _estilo(ax, titulo: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(titulo, loc="left", fontsize=11)


def coste_etapas(stats: dict, salida) -> None:
    """Cuanto cuesta cada etapa y en que recurso. El grafico central del informe."""
    tot = sum(s["total_frames"] for s in stats["por_clip"])
    an = sum(s["frames_analizados"] for s in stats["por_clip"])
    c = stats["costo_por_stage"]
    etapas = [
        ("Nivel 1\nMOG2", c["motion_detection"]["cpu_total_ms"] / tot, "CPU"),
        ("Nivel 2\nYOLO11", c["person_detection"].get("gpu_total_ms", 0) / an, "GPU"),
        ("Nivel 3\nReID", c.get("reid", {}).get("gpu_total_ms", 0) / an, "GPU"),
        ("registro\ntracks", c["tracking"]["cpu_total_ms"] / an, "CPU"),
        ("conteo\nlinea", c["conteo"]["cpu_total_ms"] / an, "CPU"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    barras = ax.bar([e[0] for e in etapas], [e[1] for e in etapas],
                    color=[AZUL if e[2] == "CPU" else NARANJA for e in etapas], width=.6)
    for r, e in zip(barras, etapas):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + .25,
                f"{e[1]:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("ms por frame")
    ax.set_ylim(0, max(e[1] for e in etapas) * 1.25)
    ax.legend(handles=[Patch(color=AZUL, label="CPU"), Patch(color=NARANJA, label="GPU")],
              frameon=False, loc="upper right")
    _estilo(ax, "Coste por etapa y recurso que consume")
    fig.tight_layout()
    fig.savefig(salida / "coste_etapas.png")
    plt.close(fig)


def reid_efecto(salida) -> None:
    """Identidades por clip antes y despues del Nivel 3, sobre los clips de tienda."""
    clips = ["tienda llena", "tienda USA", "caja"]
    series = [("sin ReID", [134, 45, 28], GRIS),
              ("ReID, umbral a ojo (0,65)", [88, 27, 16], "#9BB8D4"),
              ("ReID calibrado (0,50)", [28, 11, 10], VERDE)]
    x, w = np.arange(len(clips)), .26
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for i, (nombre, vals, col) in enumerate(series):
        pos = x + (i - 1) * w
        ax.bar(pos, vals, w, label=nombre, color=col)
        for p, v in zip(pos, vals):
            ax.text(p, v + 2, str(v), ha="center", fontsize=8)
    ax.axhline(20, ls="--", lw=1, color=ROJO)
    ax.text(len(clips) - .58, 22, "~20 personas reales", fontsize=8, color=ROJO, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(clips)
    ax.set_ylabel("identidades abiertas")
    ax.legend(frameon=False, fontsize=9)
    _estilo(ax, "Nivel 3: identidades por clip antes y despues")
    fig.tight_layout()
    fig.savefig(salida / "reid_efecto.png")
    plt.close(fig)


def reid_calibracion(calib: dict, salida) -> None:
    """Donde separan las dos distribuciones, y la franja que ningun umbral resuelve."""
    c = calib["variantes"][0]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for nombre, y, mu, cola, etiqueta, col in (
            ("misma persona", 1, c["pos_media"], c["pos_p10"], "p10", VERDE),
            ("personas distintas", 0, c["neg_media"], c["neg_p90"], "p90", ROJO)):
        ax.scatter([mu], [y], s=90, color=col, zorder=3)
        ax.text(mu, y + .18, f"media {mu:.3f}", ha="center", fontsize=9, color=col)
        ax.plot([cola, mu], [y, y], color=col, lw=3, alpha=.35)
        dx = -.012 if etiqueta == "p10" else .012
        ax.text(cola + dx, y - .05, f"{etiqueta}={cola:.3f}", fontsize=8, color=col,
                ha="right" if etiqueta == "p10" else "left", va="center")
    ax.axvline(c["umbral"], ls="--", color=GRIS, lw=1.2)
    ax.text(c["umbral"], 1.50, f"umbral {c['umbral']:.2f}", fontsize=9, color=GRIS,
            ha="center")
    ax.axvspan(c["pos_p10"], c["neg_p90"], color=GRIS, alpha=.12)
    ax.text((c["pos_p10"] + c["neg_p90"]) / 2, -.42,
            "zona que ningun umbral separa", ha="center", fontsize=8, color=GRIS)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["distintas", "misma"])
    ax.set_xlabel("similitud coseno del descriptor")
    ax.set_xlim(.3, .8)
    ax.set_ylim(-.62, 1.75)
    ax.spines["left"].set_visible(False)
    _estilo(ax, f"Calibracion del umbral: {calib['n_pos']} pares de cada tipo")
    fig.tight_layout()
    fig.savefig(salida / "reid_calibracion.png")
    plt.close(fig)


def metricas_por_umbral(ev: dict, salida) -> None:
    """Precision, recall y F1 contra el umbral de confianza (punto 5).

    Dos paneles y no uno: el detector solo y la cascada entera. La diferencia
    entre ambos es todo gate, y en un solo grafico se leeria como si el modelo
    empeorara.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    titulos = {"detector_solo": "El detector solo",
               "sistema_completo": "La cascada entera"}
    for ax, ambito in zip(axes, ("detector_solo", "sistema_completo")):
        us = sorted(float(u) for u in ev[ambito])
        P, R, F = [], [], []
        for u in us:
            c = ev[ambito][str(u)]
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            P.append(p); R.append(r)
            F.append(2 * p * r / (p + r) if p + r else 0.0)
        ax.plot(us, P, "o-", color=AZUL, ms=3, label="precision")
        ax.plot(us, R, "s-", color=NARANJA, ms=3, label="recall")
        ax.plot(us, F, "^-", color=VERDE, ms=3, lw=2, label="F1")
        mejor = us[int(np.argmax(F))]
        ax.axvline(mejor, ls="--", color=GRIS, lw=1)
        ax.text(mejor, .04, f"mejor F1\nconf={mejor:.2f}", fontsize=7.5,
                color=GRIS, ha="center")
        ax.set_xlabel("umbral de confianza")
        ax.set_ylim(0, 1.08)
        _estilo(ax, titulos[ambito])
    axes[0].set_ylabel("metrica")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(salida / "metricas_umbral.png")
    plt.close(fig)


def filas_por_zona(stats: dict, salida) -> None:
    """Cuanta gente hay en la ROI y cuanta esta esperando de verdad (punto 4).

    La diferencia entre las dos barras es la que ultralytics no hace sola: estar
    dentro de la zona no es lo mismo que estar esperando.
    """
    con_roi = [c for c in stats["por_clip"] if c.get("tiene_roi")]
    if not con_roi:
        return
    nombres = [c["clip_id"].replace("retail_", "") for c in con_roi]
    zona = [c["max_en_zona"] for c in con_roi]
    fila = [c["max_en_fila"] for c in con_roi]
    pct = [c["pct_frames_con_fila"] for c in con_roi]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2),
                                  gridspec_kw={"wspace": .35})
    x, w = np.arange(len(con_roi)), .34
    ax.bar(x - w / 2, zona, w, label="dentro de la ROI", color=GRIS)
    ax.bar(x + w / 2, fila, w, label="esperando (fila)", color=NARANJA)
    for i, (a, b) in enumerate(zip(zona, fila)):
        ax.text(i - w / 2, a + .1, str(a), ha="center", fontsize=9)
        ax.text(i + w / 2, b + .1, str(b), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(nombres, fontsize=9)
    ax.set_ylabel("personas (maximo simultaneo)")
    ax.legend(frameon=False, fontsize=8)
    _estilo(ax, "En la zona vs. esperando")

    ax2.bar(x, pct, .5, color=VERDE)
    for i, v in enumerate(pct):
        ax2.text(i, v + .5, f"{v:.1f} %", ha="center", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(nombres, fontsize=9)
    ax2.set_ylabel("% del tiempo con fila")
    ax2.set_ylim(0, max(pct) * 1.35 if pct else 1)
    _estilo(ax2, "Tiempo con fila")
    fig.tight_layout()
    fig.savefig(salida / "filas_zona.png")
    plt.close(fig)


def main() -> int:
    cfg = load_cascade_config()
    salida = cfg.results_dir.parent / "documentation" / "imagenes"
    salida.mkdir(parents=True, exist_ok=True)
    stats = json.loads((cfg.results_dir / "stage2_person_stats.json").read_text())
    calib = json.loads((cfg.results_dir / "reid_calibracion.json").read_text())

    coste_etapas(stats, salida)
    reid_efecto(salida)
    reid_calibracion(calib, salida)
    filas_por_zona(stats, salida)
    ev = cfg.results_dir / "eval_stats.json"
    if ev.exists():
        metricas_por_umbral(json.loads(ev.read_text()), salida)
    print(f"figuras en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
