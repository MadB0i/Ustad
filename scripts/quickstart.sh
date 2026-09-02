#!/usr/bin/env bash
# Ustad quickstart (Linux / macOS).
#
#   ./scripts/quickstart.sh                 set up, then launch the app
#   ./scripts/quickstart.sh --cpu           force the CPU-only torch build
#   ./scripts/quickstart.sh --check         set up, then run the hardware preflight
#   ./scripts/quickstart.sh --skip-install  skip dependency install and just launch
#
# Nothing here reaches the network except pip and (optionally) the one-time model
# downloads. The Ollama daemon is started if it is not already listening; it is never
# stopped or killed, because it may be serving something else.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FORCE_CPU=0
RUN_CHECK=0
SKIP_INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --cpu) FORCE_CPU=1 ;;
    --check) RUN_CHECK=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '\n== %s\n' "$1"; }
note() { printf '   %s\n' "$1"; }
die()  { printf '\nx  %s\n' "$1" >&2; exit 1; }

# --- 1. interpreter ------------------------------------------------------------------
# torch publishes cp312 wheels; a 3.13/3.14 interpreter will fail to resolve them, so
# the version is checked here rather than discovered halfway through a 3 GB download.

say 'Locating Python 3.12'
PYTHON=''
for candidate in python3.12 python3.11 python3.10 python3 python; do
  command -v "$candidate" >/dev/null 2>&1 || continue
  version="$("$candidate" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)"
  case "$version" in
    3.12|3.11|3.10) PYTHON="$candidate"; note "$candidate -> Python $version"; break ;;
  esac
done
[ -n "$PYTHON" ] || die 'No Python 3.10-3.12 found. Install Python 3.12 (torch has no wheels for 3.13+ yet).'

# --- 2. virtual environment ----------------------------------------------------------

VENV="$ROOT/.venv"
VPY="$VENV/bin/python"

if [ ! -x "$VPY" ]; then
  say 'Creating .venv'
  "$PYTHON" -m venv "$VENV"
  [ -x "$VPY" ] || die 'venv creation failed.'
else
  say 'Reusing existing .venv'
fi

# --- 3. dependencies ----------------------------------------------------------------

if [ "$SKIP_INSTALL" -eq 0 ]; then
  INDEX=''
  if [ "$FORCE_CPU" -eq 1 ]; then
    note 'CPU build requested'
    INDEX='https://download.pytorch.org/whl/cpu'
  elif [ "$(uname -s)" = "Darwin" ]; then
    # Apple silicon gets MPS from the default wheel; there is no CUDA and no
    # bitsandbytes, so 4-bit is unavailable and the CPU/MPS preset is used.
    note 'macOS: installing the default wheel (MPS, no 4-bit quantisation)'
  elif command -v nvidia-smi >/dev/null 2>&1; then
    note "detected: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    # cu126 is the most recent index that still ships kernels for every arch from
    # Maxwell through Hopper, which includes Turing (sm_75).
    INDEX='https://download.pytorch.org/whl/cu126'
  else
    note 'no NVIDIA GPU detected; installing the CPU build'
    INDEX='https://download.pytorch.org/whl/cpu'
  fi

  say 'Installing torch'
  "$VPY" -m pip install --disable-pip-version-check --upgrade pip
  if [ -n "$INDEX" ]; then
    "$VPY" -m pip install --disable-pip-version-check torch --index-url "$INDEX"
  else
    "$VPY" -m pip install --disable-pip-version-check torch
  fi

  say 'Installing the rest of the requirements'
  "$VPY" -m pip install --disable-pip-version-check -r "$ROOT/requirements.txt"
fi

# --- 4. Ollama ----------------------------------------------------------------------

say 'Checking the Ollama daemon'
probe() { "$VPY" - <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=3) as r:
        print(json.load(r).get("version", "?"))
except Exception:
    sys.exit(1)
PY
}

if version="$(probe 2>/dev/null)"; then
  note "already running: $version"
  LISTENING=1
elif command -v ollama >/dev/null 2>&1; then
  note 'not listening; starting `ollama serve` in the background'
  nohup ollama serve >/dev/null 2>&1 &
  LISTENING=0
  for _ in $(seq 1 20); do
    sleep 0.5
    if version="$(probe 2>/dev/null)"; then note "daemon is up: $version"; LISTENING=1; break; fi
  done
  [ "$LISTENING" -eq 1 ] || note 'daemon did not come up in 10s; start it yourself with: ollama serve'
else
  note 'ollama is not on PATH. Install it from https://ollama.com, then: ollama pull qwen2.5:3b'
  LISTENING=0
fi

if [ "${LISTENING:-0}" -eq 1 ]; then
  tags="$("$VPY" - <<'PY'
import json, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as r:
        print(" ".join(m.get("name", "") for m in json.load(r).get("models", []) if m.get("name")))
except Exception:
    print("")
PY
)"
  if [ -n "$tags" ]; then
    note "teachers available: $tags"
  else
    note 'no models pulled yet. A good 4 GB-friendly teacher: ollama pull qwen2.5:3b'
  fi
fi

# --- 5. go --------------------------------------------------------------------------

if [ "$RUN_CHECK" -eq 1 ]; then
  say 'Running the hardware preflight'
  exec "$VPY" "$ROOT/scripts/smoke_test.py" --check
fi

say 'Starting Ustad on http://127.0.0.1:8177'
note 'Ctrl+C to stop. Everything is written under ./data'
exec "$VPY" -m backend.server
