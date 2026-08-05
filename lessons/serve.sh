#!/usr/bin/env bash
# Serve the lesson viewers locally.
# Usage: ./serve.sh  (then open http://localhost:8080)
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8080}"
echo "Serving lessons/ at http://localhost:$PORT  (Ctrl-C to stop)"
exec python3 -m http.server "$PORT"
