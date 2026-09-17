"""Tests del Nivel 3 (re-identificacion por apariencia).

El modelo pesa y necesita GPU, asi que se inyecta un embebedor falso: lo que se
testea es la LOGICA de reenganche —a quien se compara, a quien no, y cuando se
deja de recordar— que es donde estan las decisiones que pueden empeorar el
seguimiento en vez de arreglarlo.
"""

from __future__ import annotations

import numpy as np

from core.perception.reid import ReIdConfig, ReIdentificador
from core.types import Detection

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _det(track_id, x=100, y=100, w=60, h=200):
    return Detection(bbox=(float(x), float(y), float(x + w), float(y + h)),
                     label="person", score=0.9, track_id=track_id)


class ReidFalso(ReIdentificador):
    """Devuelve un vector fijo por 'persona', segun el color que se le indique.

    Asi se puede simular 'esta persona se ve igual que aquella' sin cargar DINOv2.
    """

    def __init__(self, apariencias, **kw):
        super().__init__(**kw)
        self.apariencias = apariencias  # track_id -> etiqueta de apariencia
        self._pendiente = []

    def _recorte(self, frame, det):
        self._pendiente.append(det.track_id)
        return np.zeros((200, 60, 3), dtype=np.uint8)

    def _embeber(self, recortes):
        vs = []
        for tid in self._pendiente[-len(recortes):]:
            a = self.apariencias.get(tid, tid)
            v = np.zeros(8, dtype=np.float32)
            v[a % 8] = 1.0  # vectores ortogonales entre apariencias distintas
            vs.append(v)
        self._pendiente = []
        self.n_embebidos += len(recortes)
        return np.stack(vs)


def _reid(apariencias=None, **cfg):
    return ReidFalso(apariencias or {}, cfg=ReIdConfig(**cfg), fps=25.0, device="cpu")


def test_una_persona_estable_mantiene_su_id():
    r = _reid()
    ids = [r.procesar(FRAME, [_det(7)])[0].track_id for _ in range(10)]
    assert len(set(ids)) == 1
    assert r.n_personas == 1


def test_un_id_nuevo_con_la_misma_apariencia_recupera_el_viejo():
    """El caso que motiva todo: la persona se tapa y ByteTrack le da un ID nuevo."""
    r = _reid(apariencias={1: 0, 2: 0})  # ByteTrack la parte en 1 y 2, misma pinta
    assert r.procesar(FRAME, [_det(1)])[0].track_id == 1
    r.procesar(FRAME, [])  # desaparece un frame
    salida = r.procesar(FRAME, [_det(2)])[0]
    assert salida.track_id == 1  # le devolvio el ID viejo
    assert r.n_personas == 1
    assert r.n_reenganches == 1
    assert salida.meta["track_id_bytetrack"] == 2  # queda registrado el original


def test_dos_apariencias_distintas_no_se_fusionan():
    r = _reid(apariencias={1: 0, 2: 3})
    r.procesar(FRAME, [_det(1)])
    r.procesar(FRAME, [])
    assert r.procesar(FRAME, [_det(2)])[0].track_id != 1
    assert r.n_personas == 2


def test_nunca_se_fusiona_con_alguien_que_esta_en_pantalla():
    """Dos personas parecidas caminando juntas: fusionarlas es el peor error."""
    r = _reid(apariencias={1: 0, 2: 0})  # se ven IGUAL y estan las dos a la vez
    salida = r.procesar(FRAME, [_det(1, x=100), _det(2, x=300)])
    assert len({d.track_id for d in salida}) == 2
    assert r.n_personas == 2


def test_pasada_la_ventana_se_da_por_persona_nueva():
    """Una oclusion muy larga cuenta como alguien nuevo: es lo que se pidio."""
    r = _reid(apariencias={1: 0, 2: 0}, ventana_s=1.0)  # 25 frames
    r.procesar(FRAME, [_det(1)])
    for _ in range(40):
        r.procesar(FRAME, [])
    assert r.procesar(FRAME, [_det(2)])[0].track_id != 1
    assert r.n_personas == 2


def test_no_se_embebe_en_cada_frame():
    """Si costara un forward por frame, el Nivel 3 saldria mas caro que el 2."""
    r = _reid(refresco_s=100.0)  # refresco practicamente apagado
    for _ in range(30):
        r.procesar(FRAME, [_det(1)])
    assert r.n_embebidos == 1  # solo la primera vez


def test_la_galeria_se_refresca_cada_tanto():
    """La apariencia cambia con el angulo: un solo vector envejece."""
    r = _reid(refresco_s=0.4)  # 10 frames
    for _ in range(30):
        r.procesar(FRAME, [_det(1)])
    assert 2 <= r.n_embebidos <= 5


def test_la_galeria_no_crece_sin_limite():
    r = _reid(refresco_s=0.04, max_por_persona=3)
    for _ in range(40):
        r.procesar(FRAME, [_det(1)])
    assert len(r._personas[1].vectores) == 3


def test_un_recorte_muy_chico_no_rompe_nada():
    """Sin textura no se puede reidentificar: se le da ID nuevo, no se adivina."""
    r = ReIdentificador(cfg=ReIdConfig(min_lado_px=50), fps=25.0, device="cpu")
    salida = r.procesar(FRAME, [_det(1, w=10, h=10)])
    assert salida[0].track_id is not None
    assert r.n_embebidos == 0  # nunca llamo al modelo


def test_detecciones_sin_track_id_pasan_sin_tocarse():
    r = _reid()
    d = Detection(bbox=(0.0, 0.0, 10.0, 10.0), label="person", score=0.5)
    assert r.procesar(FRAME, [d])[0].track_id is None


def test_reset_vacia_la_galeria():
    r = _reid()
    r.procesar(FRAME, [_det(1)])
    r.reset()
    assert r.n_personas == 0


# ---- uso de la mascara del detector ----------------------------------------

def _con_mascara(frame_h=480, frame_w=640):
    """Una deteccion cuya mascara cubre solo la mitad izquierda de su caja."""
    d = _det(1, x=100, y=100, w=200, h=300)
    m = np.zeros((frame_h // 2, frame_w // 2), dtype=np.float32)  # media resolucion
    m[50:200, 50:100] = 1.0
    d.mask = m
    return d


def test_la_mascara_borra_el_fondo_del_recorte():
    """Sin esto, dos personas frente a la misma gondola se parecen POR la gondola."""
    r = ReIdentificador(cfg=ReIdConfig(usar_mascara=True), fps=25.0, device="cpu")
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)  # fondo claro uniforme
    sub = r._recorte(frame, _con_mascara())
    assert sub is not None
    assert (sub == 0).any()   # hay fondo borrado
    assert (sub > 0).any()    # y persona conservada


def test_sin_mascara_el_recorte_va_entero():
    r = ReIdentificador(cfg=ReIdConfig(usar_mascara=False), fps=25.0, device="cpu")
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    assert (r._recorte(frame, _con_mascara()) == 200).all()


def test_una_deteccion_sin_mascara_no_rompe():
    """YOLO-World no es `-seg`: no hay mascara y el recorte tiene que salir igual."""
    r = ReIdentificador(cfg=ReIdConfig(usar_mascara=True), fps=25.0, device="cpu")
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    d = _det(1)
    assert d.mask is None
    assert (r._recorte(frame, d) == 200).all()


def test_una_mascara_demasiado_chica_cae_al_recorte_crudo():
    """Mejor fondo de mas que borrar a la persona entera.

    La mascara de _con_mascara cubre la mitad del recorte, asi que con un
    cobertura_min de 0.9 se la considera rota y se usa el rectangulo crudo.
    """
    r = ReIdentificador(cfg=ReIdConfig(usar_mascara=True, cobertura_min=0.9),
                        fps=25.0, device="cpu")
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    assert (r._recorte(frame, _con_mascara()) == 200).all()


def test_una_mascara_con_cobertura_suficiente_si_se_usa():
    """El otro lado del umbral, para que quede fijado donde corta."""
    r = ReIdentificador(cfg=ReIdConfig(usar_mascara=True, cobertura_min=0.2),
                        fps=25.0, device="cpu")
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    assert (r._recorte(frame, _con_mascara()) == 0).any()
