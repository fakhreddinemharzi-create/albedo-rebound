#!/usr/bin/env bash
# Extensions of the reference (v3) chain: lower-boundary test, projected-climate
# summary, street canyons, the contiguous United States, figures, numbers and
# the manuscript.  Run after run_v3.sh / run_v3b.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
step() { echo "=== $(date -u +%H:%M:%S) $*"; }
step boundary;        $PY scripts/22_boundary_test.py --workers 3
step climate-summary; $PY scripts/51_climate_summary.py
step canyon;          $PY scripts/64_canyon.py
step canyon-mc;       $PY scripts/65_canyon_mc.py
step conus-kernel;    $PY scripts/66_conus_kernel.py
step conus-ledger;    $PY scripts/67_conus_ledger.py
step conus-states;    $PY scripts/68_conus_states.py
step figures
for f in sites verification evaluation thermal spacetime3d structural strain3d frostmech kernel time ledger design \
         manifold3d conus season canyon canyon_fate uncertainty climate boundary street3d graphical_abstract; do
  $PY scripts/fig_${f}.py
done
step numbers;         $PY scripts/70_numbers.py
step manuscript;      make -C manuscript
step done
