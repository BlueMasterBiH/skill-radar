#!/usr/bin/env bash
# One-time setup for arcads-cut. Safe to run again: prints READY when everything is in place.
set -e
cd "$(dirname "$0")/.."
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "NEED_FFMPEG: install ffmpeg (Mac: brew install ffmpeg, Windows: winget install ffmpeg)"; exit 1
fi
if [ -x .venv/bin/python ] && .venv/bin/python -c "import mediapipe, cv2, PIL, numpy, faster_whisper" 2>/dev/null; then echo READY; exit 0; fi
PY=""
for v in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$v" >/dev/null 2>&1 && "$v" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then PY="$v"; break; fi
done
[ -n "$PY" ] || { echo "NEED_PYTHON: install Python 3.10-3.13 (python.org), then run this again"; exit 1; }
"$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python -c "import mediapipe, cv2, PIL, numpy, faster_whisper; print('READY')"
