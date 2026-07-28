#!/bin/bash
# =====================================================================================
# Nmin/Nmax INTERVAL ABLATION (Table 5) — the full (Nmin, Nmax) grid, serially.
# Each point = ACAS-SI contrib (target 0.986, MIMD-PID) with the adaptive interval clamped
# to [Nmin, Nmax], run on gsm8k/mbpp/bbh by run_interval_point.sh. To parallelize points
# across GPUs, call run_interval_point.sh directly with its GPU/SLOT arguments instead.
#
#   bash sweep_interval.sh
# =====================================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# (Nmin Nmax) grid: symmetric lower/higher on each axis around the default 50/500 —
# isolates each bound's effect on accuracy.
GRID=(
  "50 500"     # default
  "25 500"     # Nmin lower (/2)
  "100 500"    # Nmin higher (x2)
  "50 250"     # Nmax lower (/2)
  "50 1000"    # Nmax higher (x2)
  "50 2000"    # Nmax stress (x4) — ceiling dominates steady-state, so probe it wide
)

echo "=== INTERVAL ABLATION: ${#GRID[@]} (Nmin,Nmax) points x {gsm8k,mbpp,bbh} ==="
for pair in "${GRID[@]}"; do
  set -- $pair
  bash "$HERE/run_interval_point.sh" "$1" "$2"
done
echo "=== INTERVAL ABLATION COMPLETE -> runs/ (accuracy in each run's results json) ==="
