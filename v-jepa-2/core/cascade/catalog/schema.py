"""Esquema del corpus de video: una entrada por clip, con los ejes de variacion
que despues explican las diferencias de costo y de accuracy entre clips.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class CameraHeight(str, Enum):
    """Altura de montaje de la camara."""

    LOW = "low"  # a la altura de la rodilla o menos
    EYE_LEVEL = "eye_level"  # a la altura de la cara
    CEILING = "ceiling"  # cenital o casi, tipico de CCTV comercial


class Lighting(str, Enum):
    """Condicion de iluminacion dominante."""

    BRIGHT = "bright"  # bien iluminado y estable
    DIM = "dim"  # poca luz, ruido de sensor alto
    MIXED = "mixed"  # zonas quemadas y zonas oscuras en el mismo frame


class CrowdDensity(str, Enum):
    """Cuanta gente hay en escena."""

    EMPTY = "empty"  # nadie
    SPARSE = "sparse"  # 1-3 personas
    BUSY = "busy"  # 4 o mas, con oclusiones


class LocationType(str, Enum):
    """Tipo de interior comercial."""

    STORE = "store"
    RESTAURANT = "restaurant"
    ENTRANCE = "entrance"
    CHECKOUT = "checkout"
    OTHER = "other"  # material interino que no es interior comercial


@dataclass
class ClipMeta:
    """Una entrada del corpus. duration_s/fps/codec salen de ffprobe, el resto se declara."""

    clip_id: str
    filename: str  # relativo a corpus/raw/
    source: str  # de donde salio: dataset publico, grabacion propia, etc
    camera_height: CameraHeight
    lighting: Lighting
    crowd_density: CrowdDensity
    location_type: LocationType
    duration_s: float
    width: int
    height: int
    fps: float
    codec: str
    n_frames: int
    sha256: str
    license: str = "unknown"  # obligatorio verificarlo antes de usar el clip
    notes: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Dict serializable a JSON (los enums salen como su string)."""
        d = asdict(self)
        for k in ("camera_height", "lighting", "crowd_density", "location_type"):
            d[k] = getattr(self, k).value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ClipMeta:
        """Reconstruye desde el dict de metadata.json."""
        d = dict(d)
        d["camera_height"] = CameraHeight(d["camera_height"])
        d["lighting"] = Lighting(d["lighting"])
        d["crowd_density"] = CrowdDensity(d["crowd_density"])
        d["location_type"] = LocationType(d["location_type"])
        return cls(**d)
