#!/bin/sh
# Keep the desktop environment separate from existing web/ingest environments.
set -eu
cd "$(dirname "$0")"
DESKTOP_PYTHON="${DESKTOP_PYTHON:-python3.11}"
if ! command -v "$DESKTOP_PYTHON" >/dev/null 2>&1; then
  echo 'Python 3.11 is required. On macOS: brew install python@3.11 ffmpeg'
  exit 1
fi
"$DESKTOP_PYTHON" -m venv .venv-desktop
.venv-desktop/bin/python -m pip install -r requirements-desktop.txt
.venv-desktop/bin/python -c 'import PySide6, cv2, torch, ultralytics; print("Desktop dependencies ready")'
echo 'Start Tengen by double-clicking Launch Tengen.command, or run .venv-desktop/bin/python tengen.py'
