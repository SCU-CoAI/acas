#!/bin/bash
# Per-token latency (Tables 3-4, Prec./L2 rows) at the PRECISION (ReLU) / L2 (SiLU) signal op-points,
# to compare against the contrib metric. Same deployed setup as run_token_timing.sh (fixed cadence,
# %=0.5) but the prec/L2 PID anchor configs from Tables 1-2 (ACAS-SI/GR prec PID, ACAS-CATS/CATS-GP
# L2 PID). Logs ptok (real per-token latency) + tok (is_update) -> joined by analyze_acas_token.py.
# Expectation: higher sparse-token cost than contrib (less aggressive pruning), ~same dense-token
# cost (recalibration is a full forward pass regardless of signal).
#   mkdir -p efficiency/results && tmux new -d -s toktimeprec 'bash efficiency/per_token_latency/run_token_timing_prec.sh > efficiency/results/toktimeprec.log 2>&1'
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
N="${N:-6}"; NG="${NG:-2048}"; GAP="${GAP:-6}"
EFF="$ROOT/efficiency"
OUT="${OUT:-$EFF/results/token_timing_prec}"; mkdir -p "$OUT"
PROMPT="$EFF/gsm8k_0shot-prompt.txt"
PROSPARSE="${PROSPARSE:-$HOME/prosparse-llama-2-7b.gguf}"
LLAMA="${LLAMA:-$HOME/llama-3.1-8b-instruct.gguf}"

run_method() {  # key dir bin "<env-prefix>" "<model+flags>"
  local key="$1" dir="$2" bin="$3" envp="$4" tail="$5"
  for r in $(seq 1 "$N"); do
    local csv="$OUT/tok_${key}_r${r}.csv"; local pcsv="$OUT/ptok_${key}_r${r}.csv"
    if [ -s "$pcsv" ]; then echo "[skip] $key r$r ($(($(wc -l < "$pcsv")-1)) tok)"; continue; fi
    echo "[run ] $key r$r @ $(date +%H:%M:%S)"
    ( cd "$dir" && env $envp ACAS_TOKEN_TIMING="$csv" ACAS_PTOK_LOG="$pcsv" $bin $tail ) > "$OUT/run_${key}_r${r}.log" 2>&1
    echo "       $(($(wc -l < "$pcsv" 2>/dev/null||echo 1)-1)) tokens, mean=$(awk -F, 'NR>1{s+=$2;n++}END{if(n)printf "%.1fms",s/n/1000}' "$pcsv" 2>/dev/null), $(awk -F, 'NR>1&&$3==1{c++}END{print c+0}' "$csv" 2>/dev/null) update"
    sleep "$GAP"
  done
}

### SI precision -- ACAS_SIGNAL=precision target 0.995, config precision_0.995.txt ###
SID="$ROOT/methods/sparseinfer"; SB="${SI_BIN:-$SID/build/bin/main}"; SA="$EFF/configs/sparseinfer"
run_method si_prec "$SID" "$SB" \
  "ACAS_SIGNAL=precision ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.995 ACAS_DENSE_UPDATE_PERCENT=0.5 ACAS_CONFIG=$SA/precision_0.995.txt" \
  "-m $PROSPARSE -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG --ignore-eos"

### Grasp precision -- target 0.997, config precision_0.997.txt ###
GD="$ROOT/methods/grasp"; GB="${GRASP_BIN:-$GD/build/bin/main}"; GA="$EFF/configs/grasp"
run_method grasp_prec "$GD" "$GB" \
  "ACAS_SIGNAL=precision ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.997 ACAS_DENSE_UPDATE_PERCENT=0.5 ACAS_CONFIG=$GA/precision_0.997.txt" \
  "-m $PROSPARSE -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG --ignore-eos"

### CATS L2 -- ACAS_SIGNAL=l2 target 0.80, config l2_0.80.txt ###
CD="$ROOT/methods/cats"; CB="${CATS_BIN:-$CD/build/bin/llama-cli}"; CC="$EFF/configs/cats"
run_method cats_l2 "$CD" "$CB" \
  "ACAS_SIGNAL=l2 ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.80 ACAS_LOG=0 ACAS_DENSE_UPDATE_PERCENT=0.5 ACAS_CONFIG=$CC/l2_0.80.txt" \
  "-m $LLAMA -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG -no-cnv --ignore-eos"

### CATS-GP L2 -- ACAS_SIGNAL=l2 target 0.80, config l2_0.80_g0.05.txt ###
PD="$ROOT/methods/cats-gp"; PB="${CATSGP_BIN:-$PD/build/bin/llama-cli}"; PC="$EFF/configs/cats-gp"
run_method catsgp_l2 "$PD" "$PB" \
  "ACAS_SIGNAL=l2 ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.80 ACAS_GATE_TARGET=0.05 ACAS_LOG=0 ACAS_DENSE_UPDATE_PERCENT=0.5 ACAS_CONFIG=$PC/l2_0.80_g0.05.txt" \
  "-m $LLAMA -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG -no-cnv --ignore-eos"

echo "=== PREC/L2 TOKEN TIMING DONE @ $(date +%H:%M:%S) ==="
ls -la "$OUT"/ptok_*.csv 2>/dev/null | awk '{print $5, $NF}'
