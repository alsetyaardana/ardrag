"""Supervises the `ardrag.mcp_server` process as a subprocess of the API process, so that
toggling MCP settings (transports, access mode, OAuth) from the web UI can restart just the MCP
server — not the whole container — and have the new settings take effect immediately.

Settings for the MCP server (transports, OAuth on/off, public URL) are read once at import time
by ardrag/mcp_server.py, baked into module-level Starlette app/route objects. There's no clean way
to hot-swap that structure within a running process, so a full restart of the subprocess is the
simplest reliable way to apply a change.
"""

import subprocess
import sys
import threading

_lock = threading.Lock()
_process: subprocess.Popen | None = None


def start() -> None:
    global _process
    with _lock:
        if _process is not None and _process.poll() is None:
            return
        _process = subprocess.Popen([sys.executable, "-m", "ardrag.mcp_server"])


def stop(timeout: float = 10.0) -> None:
    global _process
    with _lock:
        if _process is None:
            return
        if _process.poll() is None:
            _process.terminate()
            try:
                _process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _process.kill()
                _process.wait(timeout=timeout)
        _process = None


def restart() -> None:
    stop()
    start()


def is_running() -> bool:
    with _lock:
        return _process is not None and _process.poll() is None
