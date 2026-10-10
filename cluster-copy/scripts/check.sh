#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q backend/app backend/tests
PYTHONPATH=backend python -m pytest backend/tests -q
if command -v npm >/dev/null 2>&1; then
  (cd frontend && npm install --ignore-scripts && npm run typecheck && npm run build)
else
  echo "npm not installed; skipped frontend dependency/build check"
fi
