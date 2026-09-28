"""Publica la camara con las detecciones dibujadas encima, en vivo.

Lee el RTSP del simulador, corre la cascada completa (Nivel 1 -> Nivel 2 ->
seguimiento) y vuelve a publicar el video anotado como OTRO path del mismo
mediamtx. Asi se mira en el navegador lo que el sistema esta viendo, sin tocar
el stream original: `/cam1` sigue siendo la camara cruda, `/anotado` es la salida.

    camara -> /cam1 -> [este proceso] -> /anotado -> navegador

Muestra tres cosas a la vez: el gate del Nivel 1 (cuando dice "sin movimiento"
el detector no corre), las cajas con su ID, y el conteo por cruce de linea.

El detector corre en GPU, pero aun asi puede no llegar a los 30 fps de la camara
cuando casi todos los frames tienen movimiento. Dos defensas:

1. La lectura corre en su propio hilo y se queda SOLO con el ultimo frame. Si el
   analisis se atrasa, se pierden frames intermedios en vez de acumular retraso.
   Un monitor que va 40 s atrasado no sirve para nada, aunque no pierda un frame.
2. `--detectar-cada` limita cada cuantos frames con movimiento se llama al
   detector. Entre medio se siguen dibujando las ultimas cajas conocidas.

Uso:
  python -m core.cascade.stage2_person.vivo
  python -m core.cascade.stage2_person.vivo --detectar-cada 3 --escala 0.75
"""

from __future__ import annotations

import argparse
import subprocess
import threading
import time

import cv2

from core.cascade.config import load_cascade_config
from core.cascade.stage1_motion.detector import MotionDetector
from core.cascade.stage2_person.runner import construir_detector
from core.cascade.stage2_person.visualizar import caja_int, color_de, dibujar_linea
from core.perception.tracker import Tracker
from core.perception.conteo import ContadorPersonas
from core.perception.filas import DetectorFila, FilaConfig, cargar_rois
from core.perception.reid import ReIdConfig, ReIdentificador

SALIDA_POR_DEFECTO = "rtsp://mediamtx:8554/anotado"
BLANCO = (255, 255, 255)
GRIS = (170, 170, 170)


class LectorVivo:
    """Lee un stream en un hilo y expone unicamente el frame mas reciente.

    cv2.VideoCapture bufferea: si el consumidor es mas lento que la camara, cada
    read() devuelve un frame cada vez mas viejo y el retraso crece sin limite.
    Aca el hilo lee a la velocidad de la camara y pisa el frame anterior, asi que
    lo que se analiza es siempre lo ultimo que paso.
    """

    def __init__(self, url: str) -> None:
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._frame = None
        self._n = 0  # cuantos frames llegaron: sirve para saber cuantos se saltearon
        self._lock = threading.Lock()
        self._corriendo = self.cap.isOpened()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        if self._corriendo:
            self._hilo.start()

    @property
    def abierto(self) -> bool:
        return self._corriendo

    def _bucle(self) -> None:
        while self._corriendo:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._n += 1

    def ultimo(self):
        """Devuelve (frame, indice_global) o (None, n) si todavia no llego ninguno."""
        with self._lock:
            if self._frame is None:
                return None, self._n
            return self._frame.copy(), self._n

    def cerrar(self) -> None:
        self._corriendo = False
        self.cap.release()


def _abrir_ffmpeg(salida: str, ancho: int, alto: int, fps: float) -> subprocess.Popen:
    """ffmpeg que toma frames BGR crudos por stdin y los publica por RTSP."""
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{ancho}x{alto}", "-r", f"{fps:.2f}",
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-g", "30",
        "-f", "rtsp", "-rtsp_transport", "tcp", salida,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


AMARILLO = (0, 215, 255)  # el mismo de la linea de conteo


def _panel_contador(frame, ahora: int, cruzaron: int, tracks: int) -> None:
    """Panel de conteo, arriba a la derecha. Se dibuja sobre el frame, in place.

    Van los TRES numeros y no uno solo, porque miden cosas distintas y el que
    parece mas obvio es el menos confiable:

    - AHORA: personas en el cuadro en este frame. Es una cuenta directa, no
      acumula error.
    - CRUZARON: personas distintas que cruzaron la linea. Es el acumulado en el
      que se puede confiar: un track que se parte pero no cruza no lo infla.
    - tracks: identidades distintas que se abrieron. Con el Nivel 3 encendido ya
      no se dispara —en la tienda llena bajo de 134 a 28 contra ~20 personas
      reales—, pero sigue siendo el mas fragil de los tres: cada oclusion que el
      ReID no reengancha suma uno.
    """
    ancho = frame.shape[1]
    x0, y0, w, h = ancho - 190, 32, 182, 78
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (24, 24, 24), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)

    cv2.putText(frame, "AHORA", (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                GRIS, 1, cv2.LINE_AA)
    cv2.putText(frame, str(ahora), (x0 + 78, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                BLANCO, 2, cv2.LINE_AA)

    cv2.putText(frame, "CRUZARON", (x0 + 10, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                GRIS, 1, cv2.LINE_AA)
    cv2.putText(frame, str(cruzaron), (x0 + 105, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX,
                0.95, AMARILLO, 2, cv2.LINE_AA)

    cv2.putText(frame, f"ids abiertos {tracks}  (sobre-cuenta)", (x0 + 10, y0 + 71),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRIS, 1, cv2.LINE_AA)


def _hud(frame, texto_izq: str, texto_der: str, linea_conteo: str = "",
         hay_fila: bool = False) -> None:
    """Dos barras: arriba el estado de la cascada, abajo el conteo. In place."""
    ancho = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (ancho, 26), (24, 24, 24), -1)
    cv2.putText(frame, texto_izq, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1,
                cv2.LINE_AA)
    (w, _), _ = cv2.getTextSize(texto_der, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, texto_der, (ancho - w - 8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRIS, 1, cv2.LINE_AA)
    if not linea_conteo:
        return
    alto = frame.shape[0]
    cv2.rectangle(frame, (0, alto - 26), (ancho, alto), (24, 24, 24), -1)
    cv2.putText(frame, linea_conteo, (8, alto - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                AMARILLO, 1, cv2.LINE_AA)
    if hay_fila:
        aviso = "FILA"
        (w, _), _ = cv2.getTextSize(aviso, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, aviso, (ancho - w - 8, alto - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)


def _dibujar(frame, dets, fresco: bool) -> None:
    """Dibuja las cajas. Si no son de este frame van por esquinas, para no mentir."""
    for det in dets:
        x1, y1, x2, y2 = caja_int(det)
        c = color_de(det.track_id)
        if fresco:
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        else:
            # Caja vieja: solo las esquinas, para que se vea que es una estimacion.
            d = max(8, (x2 - x1) // 5)
            for (x, sx) in ((x1, 1), (x2, -1)):
                for (y, sy) in ((y1, 1), (y2, -1)):
                    cv2.line(frame, (x, y), (x + sx * d, y), c, 2)
                    cv2.line(frame, (x, y), (x, y + sy * d), c, 2)
        etiqueta = (f"id{det.track_id} {det.score:.2f}" if det.track_id is not None
                    else f"{det.label} {det.score:.2f}")
        cv2.putText(frame, etiqueta, (x1, max(38, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description="Publica el stream anotado en vivo")
    ap.add_argument("--url", default=None, help="stream de entrada (por defecto el de la config)")
    ap.add_argument("--salida", default=SALIDA_POR_DEFECTO)
    ap.add_argument("--detectar-cada", type=int, default=1,
                    help="correr el detector 1 de cada N frames con movimiento")
    ap.add_argument("--escala", type=float, default=1.0,
                    help="reescala el frame antes de analizar (0.75 = 25%% menos pixeles)")
    ap.add_argument("--fps", type=float, default=15.0, help="fps del stream de salida")
    ap.add_argument("--segundos", type=float, default=0.0, help="cortar solo tras N s (0 = infinito)")
    ap.add_argument("--roi-de", default="",
                    help="clip_id cuya zona de espera usar (de corpus/rois.json)")
    args = ap.parse_args()

    cfg = load_cascade_config()
    cv2.setNumThreads(cfg.cv_num_threads)
    url = args.url or cfg.rtsp_url

    print(f"leyendo  {url}")
    lector = LectorVivo(url)
    if not lector.abierto:
        print(f"ABORTA: no se pudo abrir {url}. Levantar el simulador:")
        print("  docker compose up -d mediamtx publisher")
        return 1

    # Esperar el primer frame para saber el tamano real de la salida.
    frame, _ = lector.ultimo()
    espera = time.monotonic() + 15.0
    while frame is None and time.monotonic() < espera:
        time.sleep(0.1)
        frame, _ = lector.ultimo()
    if frame is None:
        print("ABORTA: el stream se abrio pero no llego ningun frame")
        lector.cerrar()
        return 1

    if args.escala != 1.0:
        frame = cv2.resize(frame, None, fx=args.escala, fy=args.escala)
    alto, ancho = frame.shape[:2]
    print(f"publicando {args.salida}  ({ancho}x{alto} @ {args.fps:g} fps)")
    print("  navegador: http://localhost:8889/anotado   (WebRTC, baja latencia)")
    print("             http://localhost:8888/anotado   (HLS, ~5 s de retraso)")

    det_mov = MotionDetector(cfg.motion)
    det_per = construir_detector(cfg)
    if hasattr(det_per, "calentar"):
        print(f"calentando la GPU: {det_per.calentar():.0f} ms la primera inferencia")
    reg = Tracker(max_age=cfg.track.max_age, min_hits=cfg.track.min_hits)
    contador = ContadorPersonas(cfg.conteo.linea, ancho, alto)
    reid = (ReIdentificador(ReIdConfig(), fps=args.fps, device=cfg.person.device)
            if cfg.reid.activo else None)
    # La ROI es de la CAMARA, no del pipeline: en vivo hay que decir cual usar.
    roi = (cargar_rois(cfg.filas.rois_path).get(args.roi_de)
           if args.roi_de and cfg.filas.activo else None)
    fila = (DetectorFila(roi, ancho, alto, fps=args.fps, cfg=FilaConfig())
            if roi else None)
    if args.roi_de and roi is None:
        print(f"AVISO: {args.roi_de} no tiene ROI en corpus/rois.json, sin filas")
    en_zona = en_fila = 0
    hay_fila = False
    prompts = list(cfg.person.prompts)
    ff = _abrir_ffmpeg(args.salida, ancho, alto, args.fps)

    cajas, fresco = [], False
    n_analizados = con_movimiento = 0
    ultimo_n = -1
    t0 = time.monotonic()
    periodo = 1.0 / args.fps
    proximo = t0

    try:
        while True:
            if args.segundos and time.monotonic() - t0 >= args.segundos:
                break
            ahora = time.monotonic()
            if ahora < proximo:
                time.sleep(min(periodo / 4, proximo - ahora))
                continue
            proximo += periodo

            frame, n = lector.ultimo()
            if frame is None or n == ultimo_n:
                continue
            salteados = n - ultimo_n if ultimo_n >= 0 else 1
            ultimo_n = n
            if args.escala != 1.0:
                frame = cv2.resize(frame, (ancho, alto))

            r = det_mov.process(frame)
            if r.is_warmup:
                estado = "aprendiendo el fondo"
            elif not r.has_motion:
                estado = "sin movimiento - Nivel 2 apagado"
                cajas, fresco = [], False
            else:
                con_movimiento += 1
                if con_movimiento % max(1, args.detectar_cada) == 0:
                    cajas = det_per.track(frame, prompts)
                    if reid is not None:
                        cajas = reid.procesar(frame, cajas)
                    cajas = reg.update(cajas)
                    contador.actualizar(frame.copy(), cajas)
                    if fila is not None:
                        e = fila.actualizar(frame.copy(), cajas)
                        en_zona, en_fila, hay_fila = e.en_zona, e.en_fila, e.hay_fila
                    fresco = True
                    n_analizados += 1
                    estado = f"Nivel 2: {len(cajas)} persona(s)"
                else:
                    fresco = False
                    estado = "Nivel 2 salteado (cajas anteriores)"

            dibujar_linea(frame, cfg, roi)
            _dibujar(frame, cajas, fresco)
            _panel_contador(frame, len(cajas), contador.contados, reg.unique_count)
            transcurrido = max(1e-6, time.monotonic() - t0)
            _hud(frame,
                 f"{estado}",
                 f"unicas {reg.unique_count} | N2 {n_analizados} fr | "
                 f"salto {salteados} | {transcurrido:.0f}s",
                 linea_conteo=(f"entran {contador.entradas}  salen {contador.salidas}  "
                               f"cruzaron {contador.contados}"
                               + (f"  |  zona {en_zona}  esperando {en_fila}"
                                  if fila is not None else "")),
                 hay_fila=hay_fila)

            try:
                ff.stdin.write(frame.tobytes())
            except (BrokenPipeError, ValueError):
                print("ABORTA: ffmpeg cerro la salida")
                break
    except KeyboardInterrupt:
        print("\ncortado por el usuario")
    finally:
        lector.cerrar()
        if ff.stdin:
            ff.stdin.close()
        ff.wait(timeout=10)

    print(f"{n_analizados} frames analizados por el Nivel 2 | "
          f"{reg.unique_count} personas unicas | "
          f"{contador.entradas} entradas, {contador.salidas} salidas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
