"""Control del simulador de camara RTSP (mediamtx + ffmpeg en Docker Compose).

Expone rtsp://localhost:8554/<stream> para que el codigo aguas abajo trate el
simulador exactamente igual que una camara real.

Uso:
  python -m core.cascade.rtsp_sim.control start|stop|restart|status|logs
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

COMPOSE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "docker"


def wait_until_ready(url: str, timeout_s: float = 30.0) -> bool:
    """True si el puerto RTSP acepta conexiones antes del timeout."""
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8554
    limite = time.monotonic() + timeout_s
    while time.monotonic() < limite:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _compose(*args: str) -> int:
    """Corre docker compose en el directorio del stack y devuelve el exit code."""
    cmd = ["docker", "compose", *args]
    return subprocess.run(cmd, cwd=COMPOSE_DIR).returncode


def main() -> int:
    accion = sys.argv[1] if len(sys.argv) > 1 else "status"
    if accion == "start":
        return _compose("up", "-d", "mediamtx", "publisher")
    if accion == "stop":
        return _compose("down")
    if accion == "restart":
        _compose("down")
        return _compose("up", "-d", "mediamtx", "publisher")
    if accion == "logs":
        return _compose("logs", "-f", "--tail", "100", "publisher")
    if accion == "status":
        return _compose("ps")
    print(f"accion desconocida: {accion} (start|stop|restart|logs|status)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
