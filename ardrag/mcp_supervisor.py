"""Supervises the `ardrag.mcp_server` process as a subprocess of the API process, so that
toggling MCP settings (transports, access mode, OAuth) from the web UI can restart just the MCP
server — not the whole container — and have the new settings take effect immediately.

Settings for the MCP server (transports, OAuth on/off, public URL) are read once at import time
by ardrag/mcp_server.py, baked into module-level Starlette app/route objects. There's no clean way
to hot-swap that structure within a running process, so a full restart of the subprocess is the
simplest reliable way to apply a change.

A background watchdog also restarts the subprocess if it ever exits on its own (unhandled
exception, OOM, crash) — without it, an unattended crash would leave MCP silently down until
someone happened to save a setting (which restarts it as a side effect) or the whole container
was restarted.
"""

import logging
import subprocess
import sys
import threading
import time

logger = logging.getLogger(__name__)

WATCHDOG_INTERVAL_SECONDS = 10

_lock = threading.RLock()
_process: subprocess.Popen | None = None
_desired_running = False
_watchdog_started = False


def start() -> None:
    global _process, _desired_running
    with _lock:
        _desired_running = True
        if _process is not None and _process.poll() is None:
            return
        _process = subprocess.Popen([sys.executable, "-m", "ardrag.mcp_server"])
    _ensure_watchdog()


def stop(timeout: float = 10.0) -> None:
    global _process, _desired_running
    with _lock:
        _desired_running = False
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


def _ensure_watchdog() -> None:
    global _watchdog_started
    with _lock:
        if _watchdog_started:
            return
        _watchdog_started = True
    threading.Thread(target=_watchdog_loop, daemon=True).start()


def _watchdog_loop() -> None:
    while True:
        time.sleep(WATCHDOG_INTERVAL_SECONDS)
        with _lock:
            if not _desired_running or _process is None:
                continue
            exit_code = _process.poll()
        if exit_code is not None:
            logger.warning("MCP subprocess exited unexpectedly (code %s) — restarting it.", exit_code)
            start()
