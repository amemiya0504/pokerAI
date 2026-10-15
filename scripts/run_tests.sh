#!/usr/bin/env bash
# Run the full test suite. Extra arguments are passed to pytest.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
exec "$PY" -m pytest "$@"
