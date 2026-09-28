"""Genera roadmap.html (estilo roadmap.sh) a partir de roadmap.json.

Uso:  python docs/roadmap/build_roadmap.py
Solo biblioteca estandar (Python >= 3.8). Editar roadmap.json y regenerar.
"""
import html
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ENTRADA = AQUI / "roadmap.json"
SALIDA = AQUI / "roadmap.html"

# ---- geometria ----
ANCHO = 1180
EJE_X = ANCHO // 2
MAIN_W = 290
SUB_W = 260
SEP_LADO = 90            # distancia entre el nodo principal y la columna lateral
GAP_SUB = 8
GAP_SECCION = 70
LINEA_H = 18
PAD_V = 11
CHARS_SUB = 27           # caracteres por linea antes de partir (fuente ~14px)
CHARS_MAIN = 30
TOP = 360                # espacio para cabecera


def partir(texto, max_chars):
    palabras, lineas, actual = texto.split(), [], ""
    for p in palabras:
        if len(actual) + len(p) + (1 if actual else 0) > max_chars and actual:
            lineas.append(actual)
            actual = p
        else:
            actual = f"{actual} {p}" if actual else p
    if actual:
        lineas.append(actual)
    return lineas or [""]


def alto(lineas):
    return PAD_V * 2 + LINEA_H * len(lineas)


def esc(s):
    return html.escape(s, quote=True)


def nodo_svg(nid, x, y, w, lineas, clase, tipo):
    h = alto(lineas)
    cx = x + w / 2
    y0 = y + PAD_V + LINEA_H * 0.78
    tspans = "".join(
        f'<tspan x="{cx:.1f}" y="{y0 + i * LINEA_H:.1f}">{esc(l)}</tspan>'
        for i, l in enumerate(lineas)
    )
    return (
        f'<g class="nodo {clase} {tipo}" data-id="{nid}" tabindex="0" role="button">'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="5"/>'
        f'<text text-anchor="middle">{tspans}</text>'
        f'<g class="check" transform="translate({x + w - 16:.1f},{y + 4:.1f})">'
        f'<circle r="9" cx="0" cy="8"/><path d="M-4 8 l3 3 l5 -6"/></g>'
        f"</g>"
    ), h


def construir(datos):
    partes, nodos_info = [], {}
    conectores, eje = [], []
    y = TOP
    main_x = EJE_X - MAIN_W / 2
    col_der = EJE_X + MAIN_W / 2 + SEP_LADO
    col_izq = EJE_X - MAIN_W / 2 - SEP_LADO - SUB_W

    for sec in datos["sections"]:
        lados = {}
        for lado in ("right", "left"):
            items = sec.get(lado, [])
            bloques = [partir(it["t"], CHARS_SUB) for it in items]
            alto_lado = sum(alto(b) for b in bloques) + GAP_SUB * max(len(bloques) - 1, 0)
            lados[lado] = (items, bloques, alto_lado)

        main_lineas = partir(sec["title"], CHARS_MAIN)
        main_h = alto(main_lineas)
        alto_sec = max(main_h, lados["right"][2], lados["left"][2])
        main_y = y + (alto_sec - main_h) / 2
        main_cy = main_y + main_h / 2
        estado = sec.get("status", "todo")

        svg, _ = nodo_svg(sec["id"], main_x, main_y, MAIN_W, main_lineas, "principal", estado)
        partes.append(svg)
        nodos_info[sec["id"]] = {"t": sec["title"], "s": estado, "d": sec.get("desc", ""), "k": "section"}
        eje.append((main_y, main_y + main_h))

        for lado, (items, bloques, alto_lado) in lados.items():
            if not items:
                continue
            x = col_der if lado == "right" else col_izq
            yy = y + (alto_sec - alto_lado) / 2
            borde_main = EJE_X + MAIN_W / 2 if lado == "right" else EJE_X - MAIN_W / 2
            borde_sub = x if lado == "right" else x + SUB_W
            for i, (it, lineas) in enumerate(zip(items, bloques)):
                nid = f'{sec["id"]}_{lado[0]}{i}'
                tipo = "herramienta" if it.get("k") == "tool" else "sub"
                svg, h = nodo_svg(nid, x, yy, SUB_W, lineas, tipo, it.get("s", "todo"))
                partes.append(svg)
                nodos_info[nid] = {"t": it["t"], "s": it.get("s", "todo"), "d": it.get("d", ""),
                                   "k": it.get("k", "topic"), "sec": sec["title"]}
                cy = yy + h / 2
                dx = (borde_sub - borde_main) * 0.5
                conectores.append(
                    f'<path class="punteado" d="M{borde_main:.1f} {main_cy:.1f} '
                    f'C{borde_main + dx:.1f} {main_cy:.1f} {borde_sub - dx:.1f} {cy:.1f} '
                    f'{borde_sub:.1f} {cy:.1f}"/>'
                )
                yy += h + GAP_SUB

        # etiqueta de seccion en el lado libre (o encima si ambos lados estan ocupados)
        etiqueta = sec.get("label")
        if etiqueta:
            if not lados["left"][0]:
                ex = EJE_X - MAIN_W / 2 - 40
                conectores.append(f'<path class="solido" d="M{EJE_X - MAIN_W / 2:.1f} {main_cy:.1f} H{ex + 8:.1f}"/>')
                partes.append(f'<text class="etiqueta" x="{ex:.1f}" y="{main_cy + 6:.1f}" text-anchor="end">{esc(etiqueta)}</text>')
            elif not lados["right"][0]:
                ex = EJE_X + MAIN_W / 2 + 40
                conectores.append(f'<path class="solido" d="M{EJE_X + MAIN_W / 2:.1f} {main_cy:.1f} H{ex - 8:.1f}"/>')
                partes.append(f'<text class="etiqueta" x="{ex:.1f}" y="{main_cy + 6:.1f}">{esc(etiqueta)}</text>')
            else:
                ancho_et = len(etiqueta) * 7.6 + 16
                partes.append(f'<rect class="fondo-et" x="{EJE_X - ancho_et / 2:.1f}" y="{main_y - 29:.1f}" width="{ancho_et:.1f}" height="24" rx="4"/>')
                partes.append(f'<text class="etiqueta chica" x="{EJE_X:.1f}" y="{main_y - 11:.1f}" text-anchor="middle">{esc(etiqueta)}</text>')

        y += alto_sec + GAP_SECCION

    # eje central entre nodos principales
    lineas_eje = [f'<path class="punteado eje" d="M{EJE_X} {TOP - 170} V{TOP - 105}"/>',
                  f'<path class="solido eje" d="M{EJE_X} {TOP - 45} V{eje[0][0]:.1f}"/>']
    for (_, fin), (ini, _) in zip(eje, eje[1:]):
        lineas_eje.append(f'<path class="solido eje" d="M{EJE_X} {fin:.1f} V{ini:.1f}"/>')
    fin_y = eje[-1][1]
    lineas_eje.append(f'<path class="punteado eje" d="M{EJE_X} {fin_y:.1f} V{fin_y + 60:.1f}"/>')

    alto_total = y + 40
    cabecera = construir_cabecera(datos)
    svg = (
        f'<svg id="mapa" viewBox="0 0 {ANCHO} {alto_total:.0f}" xmlns="http://www.w3.org/2000/svg">'
        + cabecera + "".join(lineas_eje) + "".join(conectores) + "".join(partes) + "</svg>"
    )
    return svg, nodos_info


def construir_cabecera(datos):
    out = [f'<text class="titulo" x="{EJE_X}" y="{TOP - 62}" text-anchor="middle">{esc(datos["title"])}</text>']
    # caja de documentos (como "Related Roadmaps")
    docs = datos.get("docs", [])
    x, y, w = 30, 30, 300
    h = 44 + 26 * len(docs)
    out.append(f'<rect class="caja" x="{x}" y="{y}" width="{w}" height="{h}" rx="4"/>')
    out.append(f'<text class="caja-t" x="{x + 18}" y="{y + 26}">Documentos del proyecto</text>')
    for i, d in enumerate(docs):
        yy = y + 50 + 26 * i
        out.append(
            f'<a href="{esc(d["path"])}"><g class="doc"><circle cx="{x + 26}" cy="{yy - 5}" r="7"/>'
            f'<path d="M{x + 22.5} {yy - 5} l2.5 2.5 l4 -5"/>'
            f'<text x="{x + 42}" y="{yy}">{esc(d["label"])}</text></g></a>'
        )
    # caja de la idea
    lineas = partir(datos.get("idea", ""), 48)
    ix, iw = ANCHO - 30 - 380, 380
    ih = 40 + 19 * len(lineas)
    out.append(f'<rect class="caja idea" x="{ix}" y="30" width="{iw}" height="{ih}" rx="4"/>')
    out.append(f'<text class="caja-t" x="{ix + 18}" y="56">La idea, destilada</text>')
    ts = "".join(f'<tspan x="{ix + 18}" y="{80 + 19 * i}">{esc(l)}</tspan>' for i, l in enumerate(lineas))
    out.append(f'<text class="idea-t">{ts}</text>')
    return "".join(out)


PLANTILLA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roadmap AMTA Vision</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Balsamiq+Sans:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
  --fondo: #ffffff; --tinta: #000000; --suave: #555555;
  --amarillo: #ffff00; --beige: #ffe599; --azul: #2b78e4; --azul-t: #ffffff;
  --hecho: #d9d9d9; --pausa: #eeeeee; --curso: #f59e0b; --ok: #16a34a;
  --panel: #fff8dc;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--fondo); color: var(--tinta);
  font-family: "Balsamiq Sans", "Comic Sans MS", system-ui, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--fondo);
  border-bottom: 1px solid #e5e5e5; padding: 10px 16px; display: flex; gap: 16px;
  align-items: center; flex-wrap: wrap; }
header h1 { font-size: 18px; margin: 0; }
header .sub { color: var(--suave); font-size: 14px; }
.progreso { margin-left: auto; display: flex; align-items: center; gap: 10px; font-size: 14px; }
.barra { width: 180px; height: 10px; border: 2px solid var(--tinta); border-radius: 6px; overflow: hidden; }
.barra i { display: block; height: 100%; width: 0; background: var(--ok); transition: width .3s; }
.leyenda { display: flex; gap: 12px; font-size: 13px; flex-wrap: wrap; }
.leyenda span::before { content: ""; display: inline-block; width: 14px; height: 14px;
  border: 2px solid #000; border-radius: 3px; vertical-align: -3px; margin-right: 5px; background: var(--c); }
main { overflow-x: auto; padding: 0 16px 40px; }
#mapa { width: 100%; min-width: 900px; max-width: 1180px; display: block; margin: 0 auto; }
.titulo { font-size: 34px; font-weight: 700; }
.etiqueta { font-size: 19px; paint-order: stroke; stroke: #fff; stroke-width: 7px; stroke-linejoin: round; }
.etiqueta.chica { font-size: 15px; fill: var(--suave); stroke: none; }
.fondo-et { fill: var(--fondo); }
.num { font-size: 18px; }
.solido { stroke: var(--azul); stroke-width: 3.5; fill: none; }
.punteado { stroke: var(--azul); stroke-width: 3; fill: none; stroke-dasharray: 1 7; stroke-linecap: round; }
.caja { fill: #fff; stroke: #000; stroke-width: 2.5; }
.caja.idea { fill: var(--panel); }
.caja-t { font-size: 15px; font-weight: 700; }
.idea-t { font-size: 14px; }
.doc circle { fill: #555; } .doc path { stroke: #fff; stroke-width: 2; fill: none; }
.doc text { font-size: 14px; } .doc:hover text { text-decoration: underline; }
.nodo { cursor: pointer; }
.nodo rect { stroke: #000; stroke-width: 2.5; }
.nodo text { font-size: 14.5px; }
.nodo.principal rect { fill: var(--amarillo); }
.nodo.principal text { font-size: 16px; }
.nodo.sub rect { fill: var(--beige); }
.nodo.herramienta rect { fill: var(--azul); stroke: var(--azul); }
.nodo.herramienta text { fill: var(--azul-t); }
.nodo:hover rect, .nodo:focus rect { filter: brightness(0.93); outline: none; }
.nodo .check { display: none; }
.nodo.done rect { fill: var(--hecho); stroke: #777; }
.nodo.done text { text-decoration: line-through; fill: #444; }
.nodo.done .check { display: block; } .check circle { fill: var(--ok); }
.check path { stroke: #fff; stroke-width: 2.2; fill: none; }
.nodo.doing rect { stroke: var(--curso); stroke-width: 4; }
.nodo.paused rect { fill: var(--pausa); stroke: #999; stroke-dasharray: 6 4; }
.nodo.paused text { fill: #777; }
aside { position: fixed; top: 0; right: 0; height: 100%; width: min(420px, 100%);
  background: #fff; border-left: 2px solid #000; transform: translateX(100%);
  transition: transform .25s; z-index: 10; padding: 22px; overflow-y: auto;
  box-shadow: -8px 0 24px rgba(0,0,0,.12); }
aside.abierto { transform: none; }
aside h2 { margin: 6px 0 4px; font-size: 22px; }
aside .sec { color: var(--suave); font-size: 13px; }
aside p { line-height: 1.5; font-size: 15px; }
aside code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-size: 13px; }
aside .cerrar { position: absolute; top: 12px; right: 14px; border: 0; background: none; font-size: 24px; cursor: pointer; }
.estados { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
.estados button { font: inherit; font-size: 14px; border: 2px solid #000; border-radius: 6px;
  padding: 6px 10px; background: #fff; cursor: pointer; }
.estados button.activo { background: #000; color: #fff; }
.nota { font-size: 12px; color: var(--suave); margin-top: 18px; }
</style>
</head>
<body>
<header>
  <h1>__TITULO__</h1><span class="sub">__SUBTITULO__ · actualizado __FECHA__</span>
  <div class="leyenda">
    <span style="--c:#ffff00">Fase</span><span style="--c:#ffe599">Tema</span>
    <span style="--c:#2b78e4">Herramienta / opción</span><span style="--c:#d9d9d9">Hecho</span>
    <span style="--c:#fff;border-color:#f59e0b">En curso</span><span style="--c:#eeeeee">En pausa</span>
  </div>
  <div class="progreso"><div class="barra"><i id="barra"></i></div><span id="pct"></span></div>
</header>
<main>__SVG__</main>
<aside id="panel" aria-live="polite">
  <button class="cerrar" aria-label="Cerrar">×</button>
  <div class="sec" id="p-sec"></div>
  <h2 id="p-t"></h2>
  <p id="p-d"></p>
  <div class="estados">
    <button data-s="todo">Pendiente</button><button data-s="doing">En curso</button>
    <button data-s="done">Hecho</button><button data-s="paused">En pausa</button>
    <button data-s="">Restablecer</button>
  </div>
  <div class="nota">El estado marcado aquí se guarda solo en este navegador. La fuente de verdad es
  <code>docs/roadmap/roadmap.json</code>: edítalo y regenera con
  <code>python docs/roadmap/build_roadmap.py</code>.</div>
</aside>
<script>
const NODOS = __NODOS__;
const CLAVE = "amta-roadmap-estados";
let override = {};
try { override = JSON.parse(localStorage.getItem(CLAVE) || "{}"); } catch (e) { override = {}; }
const ESTADOS = ["todo", "doing", "done", "paused"];
const estadoDe = id => override[id] || NODOS[id].s;
function pintar() {
  let total = 0, hechos = 0;
  document.querySelectorAll(".nodo").forEach(g => {
    const id = g.dataset.id, s = estadoDe(id);
    ESTADOS.forEach(e => g.classList.toggle(e, e === s));
    if (NODOS[id].k !== "section" && s !== "paused") { total++; if (s === "done") hechos++; }
  });
  const pct = total ? Math.round(100 * hechos / total) : 0;
  document.getElementById("barra").style.width = pct + "%";
  document.getElementById("pct").textContent = hechos + " de " + total + " hechos (" + pct + "%)";
}
const panel = document.getElementById("panel");
let actual = null;
function md(s) {
  const d = document.createElement("div"); d.textContent = s;
  return d.innerHTML.replace(/`([^`]+)`/g, "<code>$1</code>");
}
function abrir(id) {
  actual = id; const n = NODOS[id];
  document.getElementById("p-sec").textContent = n.k === "section" ? "Fase" : (n.sec || "");
  document.getElementById("p-t").textContent = n.t;
  document.getElementById("p-d").innerHTML = md(n.d || "Sin descripción.");
  panel.querySelectorAll(".estados button").forEach(b =>
    b.classList.toggle("activo", b.dataset.s === estadoDe(id)));
  panel.classList.add("abierto");
}
document.querySelectorAll(".nodo").forEach(g => {
  g.addEventListener("click", () => abrir(g.dataset.id));
  g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); abrir(g.dataset.id); } });
});
panel.querySelector(".cerrar").onclick = () => panel.classList.remove("abierto");
document.addEventListener("keydown", e => { if (e.key === "Escape") panel.classList.remove("abierto"); });
panel.querySelectorAll(".estados button").forEach(b => b.onclick = () => {
  if (!actual) return;
  if (b.dataset.s) override[actual] = b.dataset.s; else delete override[actual];
  try { localStorage.setItem(CLAVE, JSON.stringify(override)); } catch (e) {}
  pintar(); abrir(actual);
});
pintar();
</script>
</body>
</html>
"""


def main():
    datos = json.loads(ENTRADA.read_text(encoding="utf-8"))
    svg, nodos = construir(datos)
    salida = (
        PLANTILLA.replace("__TITULO__", esc(datos["title"]))
        .replace("__SUBTITULO__", esc(datos.get("subtitle", "")))
        .replace("__FECHA__", esc(datos.get("updated", "")))
        .replace("__SVG__", svg)
        .replace("__NODOS__", json.dumps(nodos, ensure_ascii=False))
    )
    SALIDA.write_text(salida, encoding="utf-8")
    print(f"{SALIDA.name}: {len(nodos)} nodos, {len(salida) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
