#!/bin/bash
# Per-token latency (Tables 3-4, contrib rows) at the CONTRIB operating point, FIXED cadence (the
# deployed contrib-PID rows of the main table). Runs each method's instrumented binary directly (NO
# power sampler, so the 25ms INA3221 polling can't perturb per-token latency) with ACAS_TOKEN_TIMING
# -> one CSV per rep (idx,dt_us,is_update). analyze_acas_token.py joins tok_/ptok_ CSVs by idx ->
# normal vs dense-update split + worst-case.
#
# p50/p95/p99 don't depend on fixed-vs-mimd, so we run ONLY fixed = the contrib op-point.
# DENSE_UPDATE_PERCENT=0.5 = deployed: ~5 dense-update (calibration) tokens per 1000 -> the tail.
#
#   mkdir -p efficiency/results && tmux new -d -s toktime 'bash efficiency/per_token_latency/run_token_timing.sh > efficiency/results/toktime.log 2>&1'
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
N="${N:-6}"; NG="${NG:-2048}"; GAP="${GAP:-6}"
EFF="$ROOT/efficiency"
OUT="${OUT:-$EFF/results/token_timing}"; mkdir -p "$OUT"
PROMPT="$EFF/gsm8k_0shot-prompt.txt"
PROSPARSE="${PROSPARSE:-$HOME/prosparse-llama-2-7b.gguf}"
LLAMA="${LLAMA:-$HOME/llama-3.1-8b-instruct.gguf}"

# run one method: key dir bin "<env-prefix>" "<model+flags>"
# Two per-rep logs: ptok_*.csv = REAL per-token latency from llama.cpp's own accumulator (idx,dt_us);
# tok_*.csv = main.cpp hook (idx, enqueue_us, is_update). Joined by idx in analyze_acas_token.py.
run_method() {
  local key="$1" dir="$2" bin="$3" envp="$4" tail="$5"
  for r in $(seq 1 "$N"); do
    local csv="$OUT/tok_${key}_r${r}.csv"; local pcsv="$OUT/ptok_${key}_r${r}.csv"
    if [ -s "$pcsv" ]; then echo "[skip] $key r$r ($(($(wc -l < "$pcsv")-1)) tok)"; continue; fi
    echo "[run ] $key r$r @ $(date +%H:%M:%S) -> $pcsv"
    ( cd "$dir" && env $envp ACAS_TOKEN_TIMING="$csv" ACAS_PTOK_LOG="$pcsv" $bin $tail ) > "$OUT/run_${key}_r${r}.log" 2>&1
    echo "       done: $(($(wc -l < "$pcsv" 2>/dev/null || echo 1)-1)) tokens, mean=$(awk -F, 'NR>1{s+=$2;n++}END{if(n)printf "%.1fms",s/n/1000}' "$pcsv" 2>/dev/null), $(awk -F, 'NR>1&&$3==1{c++}END{print c+0}' "$csv" 2>/dev/null) update"
    sleep "$GAP"
  done
}

### SI — contrib pid fixed 0.986 ###
SID="$ROOT/methods/sparseinfer"; SB="${SI_BIN:-$SID/build/bin/main}"; SA="$EFF/configs/sparseinfer"
run_method si "$SID" "$SB" \
  "ACAS_SIGNAL=contrib ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.986 ACAS_DENSE_UPDATE_PERCENT=0.5 ACAS_CONFIG=$SA/contrib_0.986.txt" \
  "-m $PROSPARSE -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG --ignore-eos"

### Grasp — contrib pid fixed 0.96 ###
GD="$ROOT/methods/grasp"; GB="${GRASP_BIN:-$GD/build/bin/main}"; GA="$EFF/configs/grasp"
run_method grasp "$GD" "$GB" \
  "ACAS_SIGNAL=contrib ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.96 ACAS_DENSE_UPDATE_PERCENT=0.5 ACAS_CONFIG=$GA/contrib_0.96.txt" \
  "-m $PROSPARSE -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG --ignore-eos"

### CATS — contrib pid fixed 0.89 ###
CD="$ROOT/methods/cats"; CB="${CATS_BIN:-$CD/build/bin/llama-cli}"; CC="$EFF/configs/cats"
run_method cats "$CD" "$CB" \
  "ACAS_SIGNAL=contrib ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.89 ACAS_LOG=0 ACAS_DENSE_UPDATE_PERCENT=0.5 ACAS_CONFIG=$CC/contrib_0.89.txt" \
  "-m $LLAMA -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG -no-cnv --ignore-eos"

### CATS-GP — contrib pid fixed 0.89 (gate 0.89) ###
PD="$ROOT/methods/cats-gp"; PB="${CATSGP_BIN:-$PD/build/bin/llama-cli}"; PC="$EFF/configs/cats-gp"
run_method catsgp "$PD" "$PB" \
  "ACAS_SIGNAL=contrib ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.89 ACAS_GATE_TARGET=0.89 ACAS_LOG=0 ACAS_DENSE_UPDATE_PERCENT=0.5 ACAS_CONFIG=$PC/contrib_0.89_g0.89.txt" \
  "-m $LLAMA -ngl 33 -f $PROMPT -c 4096 -b 1024 -n $NG -no-cnv --ignore-eos"

echo "=== TOKEN-TIMING DONE @ $(date +%H:%M:%S) ==="
ls -la "$OUT"/tok_*.csv 2>/dev/null | awk '{print $5, $NF}'
