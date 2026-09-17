"""Re-identificacion por apariencia: que un ID sobreviva a la oclusion.

EL PROBLEMA. ByteTrack asocia por posicion. Mientras la persona se ve, funciona;
cuando se tapa detras de una gondola y reaparece caminando, la caja ya no esta
donde el tracker la esperaba y le abre un ID nuevo. Medido sobre los clips de
tienda: 207 IDs para un puñado de personas reales. Alargar `track_buffer` de 30 a
240 frames solo baja a 181 (-13 %), porque el problema no es cuanto se espera
sino que no hay con que reconocerla.

LA SOLUCION. Guardar como se VE cada persona y reconocerla por eso. Se recorta la
caja, se la pasa por DINOv2 y queda un vector de 384 dimensiones. Cuando el
tracker abre un ID nuevo, se compara ese vector contra los de las personas que se
perdieron hace poco; si se parece lo suficiente, se le devuelve el ID viejo.

POR QUE ES BARATO. El embedding NO corre en cada frame: corre cuando aparece un ID
que no se vio antes, y cada tanto para refrescar la galeria. Es la misma idea del
cascade —la etapa cara solo se ejecuta cuando hace falta— aplicada un nivel mas
abajo. En cada frame, lo normal es que no haya IDs nuevos y esto no cueste nada.

DOS REGLAS QUE EVITAN EMPEORAR LAS COSAS:

1. Solo se compara contra personas PERDIDAS, nunca contra las que estan en pantalla
   ahora mismo. Si no, dos personas parecidas que caminan juntas se fusionarian en
   una sola, que es un error peor que partir a una en dos.
2. Hay una ventana de tiempo. Pasada, la persona se da por ida y vuelve como
   nueva. Reconocer a alguien que se fue hace diez minutos no es lo que se pide y
   ademas la galeria creceria sin limite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.types import Detection

# DINOv2 y no DINOv3: el v3 esta gated en HuggingFace (`gated: manual`, hay que
# pedirle acceso a Meta y que lo aprueben). Misma familia y misma API AutoModel.
# Con acceso, basta AMTA_REID_MODELO=facebook/dinov3-vits16-pretrain-lvd1689m y un
# HF_TOKEN; el resto del pipeline no cambia. Puede importar para el resultado: la
# mejora de v3 es justo en dense features, que es lo que usa la re-identificacion.
MODELO_POR_DEFECTO = os.environ.get("AMTA_REID_MODELO", "facebook/dinov2-small")


@dataclass
class ReIdConfig:
    """Parametros del Nivel 3."""

    modelo: str = MODELO_POR_DEFECTO
    # Similitud coseno minima para dar por buena una re-identificacion.
    #
    # 0.50 NO es a ojo: sale de core.cascade.eval.calibrar_reid sobre los clips de
    # tienda (600 pares de la misma persona contra 600 de personas distintas,
    # armados sin etiquetar nada). Antes estaba en 0.65, elegido a dedo, y era
    # DEMASIADO ALTO: rechazaba reenganches buenos.
    #
    # Con 0.50: recupera el 85.7 % de las mismas personas y rechaza el 84.8 % de
    # las distintas. Ese ~85 % es un techo del descriptor, no del umbral: las colas
    # se cruzan (pos_p10=0.451 contra neg_p90=0.531), asi que hay una franja que
    # ningun corte separa. Subirlo cambia que error se comete, no cuantos.
    umbral: float = 0.50
    # Cuanto tiempo se recuerda a alguien que se fue. Pasado esto vuelve como
    # persona nueva, que para una oclusion larga es la respuesta correcta.
    ventana_s: float = 30.0
    # Cuantos vectores se guardan por persona. Varios porque la apariencia cambia
    # con el angulo; se compara contra el mejor de todos.
    max_por_persona: int = 5
    # Cada cuanto se refresca la galeria de una persona que sigue en pantalla.
    refresco_s: float = 2.0
    # Recortes mas chicos que esto no se embeben: no hay textura para reconocer.
    min_lado_px: int = 24
    # Borrar el fondo con la mascara del detector antes de embeber. El detector es
    # `-seg`, asi que la mascara ya viene calculada y hasta ahora se tiraba. Importa
    # porque el recorte es un rectangulo: dos personas distintas delante de la misma
    # gondola comparten medio recorte y se parecen POR LA GONDOLA.
    # Medido, y salio al reves de lo esperado: borrar el fondo EMPEORA la
    # separacion (0.298 -> 0.278). Sube la similitud entre la misma persona
    # (0.676 -> 0.686) pero sube mas entre personas distintas (0.378 -> 0.407),
    # probablemente porque el negro compartido de todos los recortes las acerca.
    # Queda disponible porque hace un intercambio real —rechaza mejor a las
    # distintas (84.8 % -> 90.3 %) a costa de recuperar menos (85.7 % -> 80.3 %)—
    # y sirve si algun dia importa mas no fusionar que no perder.
    usar_mascara: bool = False
    # Si la mascara cubre menos que esto del recorte, esta rota y se usa el recorte
    # crudo: mejor fondo de mas que borrar a la persona.
    cobertura_min: float = 0.05


@dataclass
class _Persona:
    """Una identidad estable, con su galeria de apariencias."""

    id_estable: int
    vectores: list = field(default_factory=list)
    ultimo_frame: int = 0
    ultimo_embed: int = -10**9


class ReIdentificador:
    """Remapea los track_id de ByteTrack a identidades estables por apariencia.

    Se cuelga DESPUES del tracker: no lo reemplaza ni lo modifica, solo corrige
    sus cortes. Si el embedding falla o el recorte es muy chico, el ID de
    ByteTrack pasa tal cual y el sistema sigue funcionando como antes.
    """

    def __init__(self, cfg: ReIdConfig | None = None, fps: float = 25.0,
                 device: str = "0") -> None:
        self.cfg = cfg or ReIdConfig()
        self.fps = max(1.0, fps)
        self.device = f"cuda:{device}" if device not in ("cpu", "") else "cpu"
        self.ventana_frames = int(round(self.cfg.ventana_s * self.fps))
        self.refresco_frames = max(1, int(round(self.cfg.refresco_s * self.fps)))
        self.gpu_time_ms = 0.0  # de la ultima llamada, lo lee el medidor
        self.n_embebidos = 0  # cuantos recortes se pasaron por el modelo
        self.n_reenganches = 0  # cuantas veces se recupero un ID
        self._proc = None
        self._modelo = None
        self.reset()

    def reset(self) -> None:
        """Vacia la galeria. Va entre clips distintos, como el reset de ByteTrack."""
        self._personas: dict[int, _Persona] = {}
        self._de_track: dict[int, int] = {}  # track_id de ByteTrack -> id estable
        self._siguiente = 1
        self._frame = -1

    # ---- modelo ------------------------------------------------------------

    def _cargar(self):
        """Carga perezosa: importar transformers arrastra medio torch."""
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._proc = AutoImageProcessor.from_pretrained(self.cfg.modelo)
        m = AutoModel.from_pretrained(self.cfg.modelo).eval()
        # half en GPU: el embedding es para comparar por coseno, no necesita fp32,
        # y a la 2060 le sobra poco de VRAM con el detector cargado al lado.
        self._modelo = m.half().to(self.device) if "cuda" in self.device else m.to(self.device)
        self._torch = torch

    def _embeber(self, recortes: list[np.ndarray]) -> np.ndarray:
        """Devuelve un vector L2-normalizado por recorte, en un solo forward."""
        import time

        if self._modelo is None:
            self._cargar()
        torch = self._torch
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            x = self._proc(images=recortes, return_tensors="pt")
            x = {k: v.to(self.device) for k, v in x.items()}
            if "cuda" in self.device:
                x = {k: (v.half() if v.is_floating_point() else v) for k, v in x.items()}
            salida = self._modelo(**x).last_hidden_state[:, 0]  # token CLS
            salida = torch.nn.functional.normalize(salida.float(), dim=1)
            if "cuda" in self.device:
                torch.cuda.synchronize()
        self.gpu_time_ms += (time.perf_counter_ns() - t0) / 1e6
        self.n_embebidos += len(recortes)
        return salida.cpu().numpy()

    # ---- galeria -----------------------------------------------------------

    def _recorte(self, frame: np.ndarray, det: Detection):
        """Recorte de la persona, con el fondo borrado si hay mascara."""
        alto, ancho = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ancho, x2), min(alto, y2)
        if x2 - x1 < self.cfg.min_lado_px or y2 - y1 < self.cfg.min_lado_px:
            return None
        sub = frame[y1:y2, x1:x2]
        if not self.cfg.usar_mascara or det.mask is None:
            return sub
        # La mascara viene en la resolucion interna del modelo (384x640), no en la
        # del frame: hay que llevarla al frame antes de recortarla.
        m = cv2.resize(det.mask.astype(np.uint8), (ancho, alto),
                       interpolation=cv2.INTER_NEAREST)[y1:y2, x1:x2]
        if float(m.mean()) < self.cfg.cobertura_min:
            return sub
        return sub * m[:, :, None]

    def _guardar(self, persona: _Persona, vector: np.ndarray) -> None:
        persona.vectores.append(vector)
        if len(persona.vectores) > self.cfg.max_por_persona:
            persona.vectores.pop(0)
        persona.ultimo_embed = self._frame

    def _mejor_candidato(self, vector: np.ndarray, activos: set[int]):
        """La persona perdida mas parecida, si supera el umbral.

        `activos` son las identidades que ya estan en pantalla en este frame: se
        excluyen a proposito. Fusionar a dos personas que se ven al mismo tiempo
        es peor que partir a una en dos.
        """
        mejor_id, mejor_sim = None, self.cfg.umbral
        for pid, p in self._personas.items():
            if pid in activos:
                continue
            if self._frame - p.ultimo_frame > self.ventana_frames:
                continue
            sim = max(float(np.dot(vector, v)) for v in p.vectores) if p.vectores else 0.0
            if sim > mejor_sim:
                mejor_id, mejor_sim = pid, sim
        return mejor_id, mejor_sim

    # ---- API ---------------------------------------------------------------

    def procesar(self, frame: np.ndarray, dets: list[Detection]) -> list[Detection]:
        """Reescribe el track_id de cada deteccion a su identidad estable."""
        self._frame += 1
        self.gpu_time_ms = 0.0
        utiles = [d for d in dets if d.track_id is not None]
        if not utiles:
            return dets

        activos = {self._de_track[d.track_id] for d in utiles
                   if d.track_id in self._de_track}

        # Un ID que nunca se vio hay que consultarlo contra la galeria; uno ya
        # conocido solo se refresca de vez en cuando. El resto no cuesta nada.
        nuevos = [d for d in utiles if d.track_id not in self._de_track]
        refrescar = [
            d for d in utiles
            if d.track_id in self._de_track
            and self._frame - self._personas[self._de_track[d.track_id]].ultimo_embed
            >= self.refresco_frames
        ]

        pendientes = nuevos + refrescar
        recortes, con_recorte = [], []
        for d in pendientes:
            r = self._recorte(frame, d)
            if r is not None:
                recortes.append(r)
                con_recorte.append(d)

        vectores = self._embeber(recortes) if recortes else np.empty((0, 0))
        por_det = {id(d): vectores[i] for i, d in enumerate(con_recorte)}

        for d in utiles:
            vec = por_det.get(id(d))
            if d.track_id in self._de_track:
                pid = self._de_track[d.track_id]
                p = self._personas[pid]
                if vec is not None:
                    self._guardar(p, vec)
            elif vec is None:
                # Sin recorte utilizable no se puede reidentificar: se le da una
                # identidad nueva en vez de adivinar.
                pid = self._nueva_persona()
                self._de_track[d.track_id] = pid
            else:
                pid, sim = self._mejor_candidato(vec, activos)
                if pid is None:
                    pid = self._nueva_persona()
                else:
                    self.n_reenganches += 1
                    d.meta["reid_sim"] = round(sim, 4)
                self._de_track[d.track_id] = pid
                self._guardar(self._personas[pid], vec)
            self._personas[pid].ultimo_frame = self._frame
            activos.add(pid)
            d.meta["track_id_bytetrack"] = d.track_id
            d.track_id = pid
        return dets

    def _nueva_persona(self) -> int:
        pid = self._siguiente
        self._siguiente += 1
        self._personas[pid] = _Persona(id_estable=pid, ultimo_frame=self._frame)
        return pid

    @property
    def n_personas(self) -> int:
        """Identidades estables abiertas desde el ultimo reset."""
        return len(self._personas)
