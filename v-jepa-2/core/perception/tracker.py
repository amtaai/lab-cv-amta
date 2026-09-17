"""Tracking multi-objeto para asociar detecciones entre frames y contar instancias.

Asigna un track_id estable a cada persona para poder contar y medir dwell-time.
Backend: ByteTrack, via ultralytics.

DONDE OCURRE LA ASOCIACION. ByteTrack no es una etapa separable: ultralytics lo
corre adentro del predictor (`model.track(persist=True, tracker="bytetrack.yaml")`)
porque necesita el frame, no solo las cajas. Asi que el track_id ya viene puesto
en la Detection cuando llega aca.

Lo que hace esta clase, entonces, es el registro: mantiene la historia por
track_id, decide cuando un track esta confirmado y responde cuantas instancias
unicas se vieron. Separarlo del detector importa porque el conteo tiene reglas
propias (min_hits, max_age) que no son las del asociador.
"""

from __future__ import annotations

from dataclasses import replace

from core.types import Detection, Track


class Tracker:
    """Registro de tracks sobre las detecciones que ByteTrack ya asocio."""

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 max_historia: int = 300):
        self.max_age = max_age  # frames sin verse antes de dar el track por ido
        self.min_hits = min_hits  # detecciones antes de contarlo como instancia real
        # Tope de la trayectoria guardada por track. Sin tope, un stream en vivo
        # crece sin limite: son horas de detecciones acumuladas. 300 a 25 fps son
        # 12 s de trayectoria, de sobra para dwell-time o reglas de zona.
        self.max_historia = max_historia
        self._tracks: dict[int, Track] = {}
        self._visto_en: dict[int, int] = {}  # track_id -> ultimo frame en que aparecio
        self._confirmados: set[int] = set()
        self._frame = -1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Registra `detections` (ya con track_id) y devuelve las confirmadas marcadas.

        Las detecciones sin track_id se dejan pasar sin registrar: son las de una
        llamada a detect() en vez de track(), y no se pueden contar sin duplicar.
        """
        self._frame += 1
        for d in detections:
            if d.track_id is None:
                continue
            t = self._tracks.get(d.track_id)
            if t is None:
                t = Track(track_id=d.track_id, label=d.label)
                self._tracks[d.track_id] = t
            # Se guarda una copia SIN la mascara. La mascara es de 384x640 float32
            # (~960 KB) y el registro no la necesita para nada: reteniendola, 25.000
            # detecciones son 24 GB y el proceso se muere. Lo mato el OOM killer una
            # vez antes de que apareciera este comentario.
            t.history.append(replace(d, mask=None))
            if len(t.history) > self.max_historia:
                del t.history[0]
            self._visto_en[d.track_id] = self._frame
            if len(t.history) >= self.min_hits:
                self._confirmados.add(d.track_id)
            d.meta["confirmado"] = d.track_id in self._confirmados
        return detections

    @property
    def unique_count(self) -> int:
        """Cantidad de instancias unicas vistas hasta ahora (solo confirmadas).

        Cuenta sobre el historico, no sobre los tracks vivos: una persona que
        entro, se la conto y se fue no tiene que desaparecer del total.
        """
        return len(self._confirmados)

    @property
    def activos(self) -> list[Track]:
        """Tracks vistos dentro de los ultimos max_age frames."""
        return [t for tid, t in self._tracks.items()
                if self._frame - self._visto_en.get(tid, -10**9) <= self.max_age]

    def historia(self, track_id: int) -> Track | None:
        """Trayectoria completa de un track, para dwell-time o analisis posterior."""
        return self._tracks.get(track_id)

    def reset(self) -> None:
        """Vacia el registro. Va junto con el reset de ByteTrack entre clips."""
        self._tracks.clear()
        self._visto_en.clear()
        self._confirmados.clear()
        self._frame = -1
