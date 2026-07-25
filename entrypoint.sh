#!/bin/sh
set -e

python -m uvicorn ardrag.api:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" &
API_PID=$!

python -m ardrag.mcp_server &
MCP_PID=$!

trap 'kill $API_PID $MCP_PID' TERM INT

wait $API_PID $MCP_PID
