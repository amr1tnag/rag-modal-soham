#!/usr/bin/env bash
# Start the app on macOS or Linux.  Usage:  ./run.sh
#
# Calls the virtual environment's python by full path, so no activation is
# needed and a fresh shell works the same as an activated one.

set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "No virtual environment found. Creating one..."
    python3 -m venv .venv
fi

# Cheap check — reinstalling every launch would add seconds to startup.
if ! "$PY" -c "import uvicorn, fastapi, numpy, ollama, pypdf" 2>/dev/null; then
    echo "Installing dependencies..."
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet -r requirements.txt
fi

if ! curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null; then
    echo "Ollama does not seem to be running."
    echo "Run 'ollama serve' in another terminal, or open the Ollama app."
    echo "Starting anyway — the sidebar will show a red dot until it is up."
    echo
fi

echo "Open http://localhost:8000  (Ctrl+C to stop)"
echo
exec "$PY" -m uvicorn app.main:app --reload
