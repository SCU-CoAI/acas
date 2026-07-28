#!/bin/bash
# =====================================================================================
# Baseline rows of Tables 1-2 (Jetson AGX Orin): TPOT / Power / Energy for
#   No Sparsity (dense)  — methods/dense-baseline (stock llama.cpp + LLAMA_FFN_ACT relu|silu toggle;
#                          the toggle is needed because ProSparse is a ReLU-FFN llama-2)
#   SparseInfer (static) — frozen at the paper config: alpha=1.03 (first 20 layers), 1.0 (rest)
#   Grasp (static)       — grasp fork FROZEN at uniform alpha=1.0
#   CATS-50% (static)    — cats fork FROZEN at the 50th-percentile calibration thresholds
# Same harness/knobs as sweep_table.sh (measure_power_latency.py, INA3221 @ 25ms, -n 2048). Run in the SAME
# session as sweep_table.sh so all rows share one thermal/clock frame.
#   mkdir -p efficiency/results && tmux new -d -s base 'bash efficiency/baselines.sh > efficiency/results/baselines.log 2>&1'
# =====================================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
N="${N:-6}"; GAP="${GAP:-8}"; NG="${NG:-2048}"
EFF="$ROOT/efficiency"
MEAS="$EFF/measure_power_latency.py"
OUT="${OUT:-$EFF/results/sweep_table.tsv}"; mkdir -p "$(dirname "$OUT")"
PROMPT="$EFF/gsm8k_0shot-prompt.txt"
PROSPARSE="${PROSPARSE:-$HOME/prosparse-llama-2-7b.gguf}"
LLAMA="${LLAMA:-$HOME/llama-3.1-8b-instruct.gguf}"
for f in "$MEAS" "$PROMPT" "$PROSPARSE" "$LLAMA"; do [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }; done

have() { awk -F'\t' -v l="$1" 'NR>1 && $1==l{c++} END{print c+0}' "$OUT" 2>/dev/null; }
do_reps() {  # label cmd
  local done; done=$(have "$1"); local need=$(( N - done ))
  if [ "$need" -le 0 ]; then echo "[skip] $1 ($done/$N done)"; return; fi
  echo "[run ] $1 — $done/$N, $need to go @ $(date +%H:%M:%S)"
  for r in $(seq 1 "$need"); do
    python3 "$MEAS" --label "$1" --interval 0.025 --tsv "$OUT" --n-gen "$NG" --cmd "$2" 2>&1 | grep -E "label=|MISSING|error" || true
    sleep "$GAP"
  done
}

SID="$ROOT/methods/sparseinfer";   SB="${SI_BIN:-$SID/build/bin/main}"
GD="$ROOT/methods/grasp";          GB="${GRASP_BIN:-$GD/build/bin/main}"
CD="$ROOT/methods/cats";           CB="${CATS_BIN:-$CD/build/bin/llama-cli}"
DB="$ROOT/methods/dense-baseline"; CLI="${DENSE_CLI:-$DB/build/bin/llama-cli}"

# --- No Sparsity (dense) ---
do_reps "DENSE prosparse" "cd $DB && env LLAMA_FFN_ACT=relu $CLI -m $PROSPARSE -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG --ignore-eos -no-cnv"
do_reps "DENSE llama31"   "cd $DB && env LLAMA_FFN_ACT=silu $CLI -m $LLAMA -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG --ignore-eos -no-cnv"

# --- SparseInfer static (alpha=1.03 first 20 layers, 1.0 rest — the paper config, frozen) ---
SIA="$EFF/results/static_si_alphas.txt"; { for i in $(seq 1 20); do echo 10300; done; for i in $(seq 21 32); do echo 10000; done; } > "$SIA"
do_reps "BASE sparseinfer" "cd $SID && env ACAS_SIGNAL=precision ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.995 ACAS_DENSE_UPDATE_PERCENT=0 ACAS_CONFIG=$SIA $SB -m $PROSPARSE -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG --ignore-eos"

# --- Grasp static (uniform alpha=1.0, frozen) ---
GRA="$EFF/results/static_grasp_alphas.txt"; yes 10000 | head -32 > "$GRA"
do_reps "BASE grasp" "cd $GD && env ACAS_SIGNAL=precision ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.997 ACAS_DENSE_UPDATE_PERCENT=0 ACAS_CONFIG=$GRA $GB -m $PROSPARSE -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG --ignore-eos"

# --- CATS-50% static (frozen calibration thresholds) ---
do_reps "BASE cats50" "cd $CD && env ACAS_SIGNAL=l2 ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.80 ACAS_LOG=0 ACAS_DENSE_UPDATE_PERCENT=0 ACAS_CONFIG=$EFF/configs/cats/static_cats50.txt $CB -m $LLAMA -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG -no-cnv --ignore-eos"

echo "=== BASELINES DONE @ $(date +%H:%M:%S) ==="
