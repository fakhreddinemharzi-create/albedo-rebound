#!/usr/bin/env bash
# Remaining steps of the reference (v3) chain after the whole-year kernel fix.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
step() { echo "=== $(date -u +%H:%M:%S) $*"; }
step sobol;              $PY scripts/60_sobol.py v3 1024
step mechanics-verify;   $PY scripts/21_verify_mechanics.py v3
step climate;            $PY scripts/50_climate_change.py v3 4
step done
