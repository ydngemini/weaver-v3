#!/usr/bin/env bash
# run.sh — start Weaver (handles spaces in path automatically)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$DIR/venv/bin/python3"

if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] venv not found at $PYTHON"
    echo "        Run: python3 -m venv \"$DIR/venv\" && \"$DIR/venv/bin/pip\" install -r requirements.txt"
    exit 1
fi

exec "$PYTHON" "$DIR/weaver.py" "$@"
