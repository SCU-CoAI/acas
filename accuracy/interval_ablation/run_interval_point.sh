#!/bin/bash
# One interval-ablation point (Table 5): ACAS-SI contrib (target 0.986, MIMD-PID) with the
# adaptive interval clamped to [Nmin, Nmax] — full gsm8k, mbpp, and bbh runs.
# ACAS_NMIN/NMAX reach the model because run_bench.sh uses `env` (not env -i).
#   Usage: run_interval_point.sh <NMIN> <NMAX> [GPU] [SLOT]
#     GPU  : CUDA index (omit on a single-GPU box)
#     SLOT : isolated .slot_<SLOT>.env (multi-GPU box; omit to use the default per-method .method_si.env)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ACC="$(cd "$HERE/.." && pwd)"
NMIN="${1:?usage: $0 <NMIN> <NMAX> [GPU] [SLOT]}"; NMAX="${2:?need NMAX}"; GPU="${3:-}"; SLOT="${4:-}"
export ACAS_NMIN="$NMIN" ACAS_NMAX="$NMAX"

# Ensure the BUNDLED editable lm_eval is active (the ProSparse BOS-token patch + the Qwen2Audio
# import fix — see setup_relu.sh; PyPI lm_eval==0.4.9 crashes on transformers 4.44.2 without it).
# Idempotent (no-op once installed).
if [ -d "$ACC/lm-evaluation-harness" ] && [ -d "$HOME/acas_venv_si" ]; then
  echo "[lm_eval] ensuring bundled editable lm_eval (Qwen2Audio gotcha fix)"
  "$HOME/acas_venv_si/bin/pip" install -q -e "$ACC/lm-evaluation-harness" >/dev/null 2>&1 || \
    echo "  WARN: bundled lm_eval install failed"
fi
GPU_ARG=();  [ -n "$GPU" ]  && GPU_ARG=(--gpu "$GPU")
SLOT_ARG=(); [ -n "$SLOT" ] && SLOT_ARG=(--slot "$SLOT")
for TASK in gsm8k mbpp bbh; do
  NAME="si_mimd_nmin${NMIN}_nmax${NMAX}_${TASK}"
  echo "=== [$(date +%H:%M:%S)] ${GPU:+GPU$GPU }${NAME} (full, target 0.986, Nmin=$NMIN Nmax=$NMAX) ==="
  bash "$ACC/run_bench.sh" --method si --target 0.986 --task "$TASK" \
       --cadence mimd --controller pid --name "$NAME" --limit 0 \
       "${GPU_ARG[@]}" "${SLOT_ARG[@]}"
done
echo "=== [$(date +%H:%M:%S)] point ${NMIN}/${NMAX} DONE -> runs/si_mimd_nmin${NMIN}_nmax${NMAX}_{gsm8k,mbpp,bbh} ==="
