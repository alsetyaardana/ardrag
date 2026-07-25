#!/bin/sh
set -e

# ardrag.api now supervises the ardrag.mcp_server subprocess itself (see ardrag/mcp_supervisor.py)
# so that toggling MCP settings from the web UI can restart just the MCP process, not the whole
# container.
exec python -m uvicorn ardrag.api:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"
