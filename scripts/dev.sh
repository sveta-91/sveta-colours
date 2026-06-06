#!/usr/bin/env bash
# Start both the static file server (port 8000) and the Python backend (port 8001).
# Foregrounds the static server, backgrounds the backend; both die on Ctrl-C.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# Backend (Flask)
(
  cd tools/image_pipeline
  source .venv/bin/activate
  python server.py
) &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT INT TERM

# Give the backend a moment to bind
sleep 0.6

echo "Backend  > http://localhost:8001  (pid $BACKEND_PID)"
echo "Frontend > http://localhost:8000/tools/photo_manager.html"
echo

# Static file server (foreground)
python3 -m http.server 8000
