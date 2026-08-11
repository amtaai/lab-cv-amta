"""Tests del control del simulador RTSP."""

from __future__ import annotations

from core.cascade.rtsp_sim.control import wait_until_ready


def test_wait_until_ready_devuelve_false_si_no_hay_servidor():
    # Puerto cerrado a proposito: debe rendirse rapido y no colgarse.
    assert wait_until_ready("rtsp://127.0.0.1:9/cam1", timeout_s=1.0) is False
