#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

if [ ! -f .venv/.deps_installed ]; then
  echo "Installing Python dependencies..."
  ./.venv/bin/pip install -r requirements.txt
  touch .venv/.deps_installed
fi

exec ./.venv/bin/python showcase.py "$@"
