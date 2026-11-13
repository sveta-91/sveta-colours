#!/usr/bin/env bash
# Start the static file server on port 8000 to serve the photo manager.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

echo "Sveta > http://localhost:8000/tools/photo_manager.html?site=sveta"
echo "Olga  > http://localhost:8000/tools/photo_manager.html?site=olga"
echo

python3 -m http.server 8000
