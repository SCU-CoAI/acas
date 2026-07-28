#!/bin/bash
# =====================================================================================
# ONE-FRAME table sweep (Jetson AGX Orin) — reproduces the TPOT / Power / Energy
# columns of Tables 1-2. Re-measures all 24 ACAS configs (4 methods x [prec/L2 PID anchor,
# prec/L2 PID*, contrib PID/PID*/RL/RL*]) in a SINGLE session so the whole block is internally
# consistent with NO normalization. -n 2048 -> a long, clean power window. Idempotent: tops each config up to N reps,
# 8s gaps. Dense + static baselines (No Sparsity / SparseInfer / Grasp / CATS-50%) are separate
# static/dense code paths, not driven by the ACAS env; see docs/REPRODUCE.md.
#
# Prereqs: each fork built (docs/INSTALL.md), models downloaded (env-overridable paths below).
#   mkdir -p efficiency/results && tmux new -d -s sweep 'bash efficiency/sweep_table.sh > efficiency/results/sweep.log 2>&1'
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
    echo "[$(date +%H:%M:%S)] $1 +$r" >> "$OUT.labels"
    python3 "$MEAS" --label "$1" --interval 0.025 --tsv "$OUT" --n-gen "$NG" --cmd "$2" 2>&1 | grep -E "label=|MISSING|error" || true
    sleep "$GAP"
  done
}
relu() { echo "cd $1 && env ACAS_SIGNAL=$4 ACAS_CONTROLLER=$5 ACAS_CADENCE=$6 ACAS_TARGET=$7 ACAS_DENSE_UPDATE_PERCENT=0.5 ACAS_CONFIG=$8 $2 -m $3 -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG --ignore-eos"; }
silu() { local p=""; [ -n "${9:-}" ] && p="ACAS_GATE_TARGET=$9 "; echo "cd $1 && env ACAS_SIGNAL=$4 ACAS_CONTROLLER=$5 ACAS_CADENCE=$6 ACAS_TARGET=$7 ${p}ACAS_LOG=0 ACAS_DENSE_UPDATE_PERCENT=0.5 ACAS_CONFIG=$8 $2 -m $3 -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG -no-cnv --ignore-eos"; }

### SI ###
SID="$ROOT/methods/sparseinfer"; SB="${SI_BIN:-$SID/build/bin/main}"; SA="$EFF/configs/sparseinfer"
do_reps "SI ANCHOR precPID" "$(relu "$SID" "$SB" "$PROSPARSE" precision pid fixed 0.995 "$SA/precision_0.995.txt")"
do_reps "SI prec PID*"      "$(relu "$SID" "$SB" "$PROSPARSE" precision pid mimd  0.995 "$SA/precision_0.995.txt")"
do_reps "SI contrib PID"  "$(relu "$SID" "$SB" "$PROSPARSE" contrib pid fixed 0.986 "$SA/contrib_0.986.txt")"
do_reps "SI contrib PID*" "$(relu "$SID" "$SB" "$PROSPARSE" contrib pid mimd  0.986 "$SA/contrib_0.986.txt")"
do_reps "SI contrib RL"   "$(relu "$SID" "$SB" "$PROSPARSE" contrib rl  fixed 0.986 "$SA/contrib_0.986.txt")"
do_reps "SI contrib RL*"  "$(relu "$SID" "$SB" "$PROSPARSE" contrib rl  mimd  0.986 "$SA/contrib_0.986.txt")"

### Grasp ###
GD="$ROOT/methods/grasp"; GB="${GRASP_BIN:-$GD/build/bin/main}"; GA="$EFF/configs/grasp"
do_reps "GR ANCHOR precPID" "$(relu "$GD" "$GB" "$PROSPARSE" precision pid fixed 0.997 "$GA/precision_0.997.txt")"
do_reps "GR prec PID*"      "$(relu "$GD" "$GB" "$PROSPARSE" precision pid mimd  0.997 "$GA/precision_0.997.txt")"
do_reps "GR contrib PID"  "$(relu "$GD" "$GB" "$PROSPARSE" contrib pid fixed 0.96 "$GA/contrib_0.96.txt")"
do_reps "GR contrib PID*" "$(relu "$GD" "$GB" "$PROSPARSE" contrib pid mimd  0.96 "$GA/contrib_0.96.txt")"
do_reps "GR contrib RL"   "$(relu "$GD" "$GB" "$PROSPARSE" contrib rl  fixed 0.96 "$GA/contrib_0.96.txt")"
do_reps "GR contrib RL*"  "$(relu "$GD" "$GB" "$PROSPARSE" contrib rl  mimd  0.96 "$GA/contrib_0.96.txt")"

### CATS ###
CD="$ROOT/methods/cats"; CB="${CATS_BIN:-$CD/build/bin/llama-cli}"; CC="$EFF/configs/cats"
do_reps "CATS ANCHOR L2PID" "$(silu "$CD" "$CB" "$LLAMA" l2 pid fixed 0.80 "$CC/l2_0.80.txt")"
do_reps "CATS L2 PID*"      "$(silu "$CD" "$CB" "$LLAMA" l2 pid mimd  0.80 "$CC/l2_0.80.txt")"
do_reps "CATS contrib PID"  "$(silu "$CD" "$CB" "$LLAMA" contrib pid fixed 0.89 "$CC/contrib_0.89.txt")"
do_reps "CATS contrib PID*" "$(silu "$CD" "$CB" "$LLAMA" contrib pid mimd  0.89 "$CC/contrib_0.89.txt")"
do_reps "CATS contrib RL"   "$(silu "$CD" "$CB" "$LLAMA" contrib rl  fixed 0.89 "$CC/contrib_0.89.txt")"
do_reps "CATS contrib RL*"  "$(silu "$CD" "$CB" "$LLAMA" contrib rl  mimd  0.89 "$CC/contrib_0.89.txt")"

### CATS-GP ###
PD="$ROOT/methods/cats-gp"; PB="${CATSGP_BIN:-$PD/build/bin/llama-cli}"; PC="$EFF/configs/cats-gp"
do_reps "CATSGP ANCHOR L2PID" "$(silu "$PD" "$PB" "$LLAMA" l2 pid fixed 0.80 "$PC/l2_0.80_g0.05.txt" 0.05)"
do_reps "CATSGP L2 PID*"      "$(silu "$PD" "$PB" "$LLAMA" l2 pid mimd  0.80 "$PC/l2_0.80_g0.05.txt" 0.05)"
do_reps "CATSGP contrib PID"  "$(silu "$PD" "$PB" "$LLAMA" contrib pid fixed 0.89 "$PC/contrib_0.89_g0.89.txt" 0.89)"
do_reps "CATSGP contrib PID*" "$(silu "$PD" "$PB" "$LLAMA" contrib pid mimd  0.89 "$PC/contrib_0.89_g0.89.txt" 0.89)"
do_reps "CATSGP contrib RL"   "$(silu "$PD" "$PB" "$LLAMA" contrib rl  fixed 0.89 "$PC/contrib_0.89_g0.89.txt" 0.89)"
do_reps "CATSGP contrib RL*"  "$(silu "$PD" "$PB" "$LLAMA" contrib rl  mimd  0.89 "$PC/contrib_0.89_g0.89.txt" 0.89)"

echo "=== SWEEP DONE @ $(date +%H:%M:%S) — $(($(wc -l < "$OUT")-1)) total rows ==="
