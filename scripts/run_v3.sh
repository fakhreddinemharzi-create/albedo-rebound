#!/usr/bin/env bash
# Downstream chain for the reference (v3) configuration: anchored deep
# boundary at 8 m, pavement roughness 0.0021 m, parametrised longwave.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
RAW=${ALRB_RAW:-data/raw}
mkdir -p "$RAW"
SW="--z0m 0.0021 --lw lw --deep dirichlet --zbot 8"
step() { echo "=== $(date -u +%H:%M:%S) $*"; }
step structural v3;      $PY scripts/40_structural.py v3
step ltpp v3;            $PY scripts/31_ltpp_benchmark.py v3
step thermal v3;         $PY scripts/36_thermal_summary.py v3
step ledger v3;          $PY scripts/62_ledger.py v3
step design v3;          $PY scripts/61_design_space.py v3
step sweeps h06 h16
$PY scripts/30_pavement_sweep.py v3_h06 --hmult 0.6 $SW --workers 2 > "$RAW/sweep_v3_h06.log" 2>&1 &
P1=$!
$PY scripts/30_pavement_sweep.py v3_h16 --hmult 1.6 $SW --workers 2 > "$RAW/sweep_v3_h16.log" 2>&1 &
P2=$!
wait $P1 $P2
step structural h06;     $PY scripts/40_structural.py v3_h06
step structural h16;     $PY scripts/40_structural.py v3_h16
step timevarying;        $PY scripts/41_timevarying_check.py v3
step sobol;              $PY scripts/60_sobol.py v3 1024
step mechanics-verify;   $PY scripts/21_verify_mechanics.py v3
step climate;            $PY scripts/50_climate_change.py v3 4
step done
