"""Fixtures compartidas de los tests del cascade."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def frame_negro():
    """Frame 640x480 BGR completamente negro."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def frame_con_rectangulo():
    """Devuelve una funcion que pinta un rectangulo blanco en (x, y)."""

    def _hacer(x: int, y: int, lado: int = 80):
        f = np.zeros((480, 640, 3), dtype=np.uint8)
        f[y : y + lado, x : x + lado] = 255
        return f

    return _hacer
