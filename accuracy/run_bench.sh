#!/bin/bash
# =====================================================================================
# ACAS accuracy runner — all four methods, one command per table row.
# Deploys the matching ACAS modeling file over the stock one, runs lm_eval, and restores
# the stock file on exit. Row = --method x --signal x --controller x --cadence.
#
# PER-METHOD ISOLATION: each method has its OWN venv (SiLU) / OWN model dir (ReLU), so the
# four rows can run in PARALLEL on one multi-GPU box without colliding on the deployed file.
# batch_size is 1 (per-forward controller) so a run can't be sped up by batching — the win is
# running rows concurrently, one per GPU. Pin a run to a GPU with --gpu N.
#
#   bash run_bench.sh --method cats   --target 0.89                       --task bbh  --gpu 0
#   bash run_bench.sh --method catsgp --target 0.89 --gate-target 0.89     --task bbh  --gpu 1
#   bash run_bench.sh --method si     --target 0.986                      --task bbh  --gpu 2
#   bash run_bench.sh --method grasp  --target 0.96                       --task bbh  --gpu 3
#   [--signal S] [--cadence fixed|mimd] [--controller pid|rl]
#   [--task gsm8k|bbh|mbpp] [--limit N] [--batch 1] [--seed N] [--name NAME] [--gpu N]
#
# --signal selects the quality metric (paper Table "Signal" column):
#   contrib (default, all methods) | si/grasp: precision (Prec. rows) | cats: activation (L2 rows)
#   | catsgp: baseline (L2 + wrong-skip rows). Cadence maps to the tables' `*`: fixed = no star,
#   mimd = starred rows. Controller maps to the PID / RL rows.
# =====================================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

METHOD=""; TASK="bbh"; TARGET=""; GATE=""; LIMIT="0"; BATCH="1"; SEED="789"; NAME=""; GPU=""; SLOT=""
CADENCE="mimd"; CONTROLLER="pid"; SIGNAL="contrib"
while [ "$#" -gt 0 ]; do case "$1" in
  --slot) SLOT="$2"; shift 2;;             # isolated per-config slot (.slot_<name>.env); else per-method
  --method) METHOD="$2"; shift 2;;
  --task) TASK="$2"; shift 2;;
  --target) TARGET="$2"; shift 2;;
  --gate-target) GATE="$2"; shift 2;;
  --cadence) CADENCE="$2"; shift 2;;       # fixed | mimd (the tables' `*` rows)
  --controller) CONTROLLER="$2"; shift 2;; # pid | rl
  --signal) SIGNAL="$2"; shift 2;;         # si/grasp: precision|l2|contrib; cats: activation|contrib; catsgp: baseline|contrib (default contrib)
  --limit) LIMIT="$2"; shift 2;;
  --batch) BATCH="$2"; shift 2;;
  --seed) SEED="$2"; shift 2;;
  --name) NAME="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 1;; esac; done
[ "$CADENCE" = aimd ] && CADENCE=mimd   # legacy alias
[ "$METHOD" = catsgr ] && METHOD=catsgp # legacy alias
[ "$CADENCE" = fixed ] || [ "$CADENCE" = mimd ] || { echo "ERROR: --cadence fixed|mimd"; exit 1; }
[ "$CONTROLLER" = pid ] || [ "$CONTROLLER" = rl ] || { echo "ERROR: --controller pid|rl"; exit 1; }

CFG="$HERE/.method_${METHOD}.env"; [ -n "$SLOT" ] && CFG="$HERE/.slot_${SLOT}.env"
[ -f "$CFG" ] || { echo "ERROR: $CFG missing — run setup_silu.sh / setup_relu.sh first"; exit 1; }
# provides: VENV, MODEL, DEPLOY, DEPLOY_CACHE (optional), TRUST ("" | ",trust_remote_code=True")
DEPLOY_CACHE=""; TRUST=""
# shellcheck disable=SC1090
source "$CFG"
[ -f "$DEPLOY.stock_bak" ] || { echo "ERROR: no stock backup at $DEPLOY.stock_bak — rerun setup"; exit 1; }

case "$TASK" in
  gsm8k) LM_TASK="gsm8k";           NFEW="--num_fewshot 8";;
  bbh)   LM_TASK="bbh_cot_fewshot"; NFEW="";;
  mbpp)  LM_TASK="mbpp";            NFEW="--num_fewshot 3"; export HF_ALLOW_CODE_EVAL=1; EXTRA="--confirm_run_unsafe_code";;
  *) echo "bad task: $TASK"; exit 1;; esac
EXTRA="${EXTRA:-}"; LIMIT_ARG=""; [ "$LIMIT" != "0" ] && LIMIT_ARG="--limit $LIMIT"

# ---- per-method signal env + staged file (by cadence) + controller + default name ----
# fixed cadence = fixed-rate dense checks; mimd = the adaptive-interval (MIMD) file.
# CONTROLLER selects pid (default) or rl (tabular Q-learning) WITHIN the chosen file.
# SIGNAL selects the quality metric WITHIN the chosen file (same env var, different value):
#   si/grasp: precision (paper Prec. rows) | l2 | contrib      cats: activation (paper L2 rows) | contrib
#   catsgp:   baseline (paper L2+wrong-skip rows) | contrib
# Default targets follow the paper operating points per signal (override with --target).
SUF=""; [ "$CADENCE" = mimd ] && SUF="_mimd"
case "$METHOD" in
  cats)
    case "$SIGNAL" in activation|contrib) ;; *) echo "ERROR: cats --signal activation|contrib"; exit 1;; esac
    [ -z "$TARGET" ] && { [ "$SIGNAL" = contrib ] && TARGET="0.89" || TARGET="0.80"; }
    SRC="$HERE/code/modeling_acas_cats${SUF}.py"
    ENVV=(ACAS_CATS_SIGNAL="$SIGNAL" ACAS_CATS_TARGET="$TARGET" ACAS_CATS_CONTROLLER="$CONTROLLER" ACAS_Q1_TAP=0)
    DESC="target=$TARGET"; DEF="cats_${CADENCE}_${CONTROLLER}_${SIGNAL}_${TARGET}_${TASK}";;
  catsgp)
    case "$SIGNAL" in baseline|contrib) ;; *) echo "ERROR: catsgp --signal baseline|contrib"; exit 1;; esac
    [ -z "$TARGET" ] && { [ "$SIGNAL" = contrib ] && TARGET="0.89" || TARGET="0.80"; }
    [ -z "$GATE" ] && GATE="0.89"
    SRC="$HERE/code/modeling_acas_catsgp${SUF}.py"
    ENVV=(ACAS_CATSGR_SIGNAL="$SIGNAL" ACAS_CATSGR_TARGET="$TARGET" ACAS_CATSGR_GATE_TARGET="$GATE" \
          ACAS_CATSGR_CONTROLLER="$CONTROLLER" ACAS_CATSGR_S_VALUES="$HERE/code/3std_averages_s_values.csv" ACAS_Q1_TAP=0)
    DESC="target=$TARGET gate=$GATE"; DEF="catsgp_${CADENCE}_${CONTROLLER}_${SIGNAL}_${TARGET}_g${GATE}_${TASK}";;
  si)
    case "$SIGNAL" in precision|l2|contrib) ;; *) echo "ERROR: si --signal precision|l2|contrib"; exit 1;; esac
    [ -z "$TARGET" ] && { [ "$SIGNAL" = contrib ] && TARGET="0.986" || TARGET="0.995"; }
    SRC="$HERE/code/modeling_acas_si${SUF}.py"
    ENVV=(ACAS_SI_SIGNAL="$SIGNAL" ACAS_SI_TARGET="$TARGET" ACAS_SI_CONTROLLER="$CONTROLLER")
    DESC="target=$TARGET"; DEF="si_${CADENCE}_${CONTROLLER}_${SIGNAL}_${TARGET}_${TASK}";;
  grasp)
    case "$SIGNAL" in precision|l2|contrib) ;; *) echo "ERROR: grasp --signal precision|l2|contrib"; exit 1;; esac
    [ -z "$TARGET" ] && { [ "$SIGNAL" = contrib ] && TARGET="0.96" || TARGET="0.997"; }
    SRC="$HERE/code/modeling_acas_grasp${SUF}.py"
    ENVV=(ACAS_GRASP_SIGNAL="$SIGNAL" ACAS_GRASP_TARGET="$TARGET" ACAS_GRASP_CONTROLLER="$CONTROLLER")
    DESC="target=$TARGET"; DEF="grasp_${CADENCE}_${CONTROLLER}_${SIGNAL}_${TARGET}_${TASK}";;
  *) echo "ERROR: --method cats|catsgp|si|grasp"; exit 1;; esac
[ -f "$SRC" ] || { echo "ERROR: staged file missing: $SRC"; exit 1; }
DESC="$DESC cadence=$CADENCE ctrl=$CONTROLLER"

[ -n "$NAME" ] || NAME="$DEF"
RESULT_DIR="$HERE/runs/$NAME"; TRACK_DIR="$RESULT_DIR/tracking"; mkdir -p "$RESULT_DIR" "$TRACK_DIR"

# ReLU (trust_remote_code): HF copies the modeling file into a dynamic-module cache on first
# load, so the cache must reflect each deploy. Re-derive the path, patch it when present, and
# restore from the model-dir stock on exit.
CACHE_FILE=""
[ -n "$TRUST" ] && CACHE_FILE="$HOME/.cache/huggingface/modules/transformers_modules/$(basename "$MODEL")/modeling_sparsellama.py"
restore() {
  echo "[restore $METHOD] stock modeling files"
  [ -f "$DEPLOY.stock_bak" ] && cp "$DEPLOY.stock_bak" "$DEPLOY"
  [ -n "$CACHE_FILE" ] && [ -f "$CACHE_FILE" ] && cp "$DEPLOY.stock_bak" "$CACHE_FILE"
  [ -d "$RESULT_DIR/sparsity_logs" ] || { [ -d "$RESULT_DIR/.cwd/sparsity_logs" ] && mv "$RESULT_DIR/.cwd/sparsity_logs" "$RESULT_DIR/sparsity_logs" 2>/dev/null || true; }
}
trap restore EXIT INT TERM
echo "[deploy $METHOD] $SIGNAL ($DESC task=$TASK gpu=${GPU:-default}) -> $DEPLOY${CACHE_FILE:+ (+cache when present)}"
cp "$SRC" "$DEPLOY"
[ -n "$CACHE_FILE" ] && [ -d "$(dirname "$CACHE_FILE")" ] && cp "$SRC" "$CACHE_FILE"

GPU_ENV=(); [ -n "$GPU" ] && GPU_ENV=(CUDA_VISIBLE_DEVICES="$GPU")
mkdir -p "$RESULT_DIR/.cwd"; cd "$RESULT_DIR/.cwd"   # SiLU controllers write sparsity_logs/ in CWD
env "${GPU_ENV[@]}" "${ENVV[@]}" ACAS_TRACK_DIR="$TRACK_DIR" PYTHONNOUSERSITE=1 \
  "$VENV/bin/python" -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,dtype=float16$TRUST" \
    --tasks "$LM_TASK" --device cuda:0 $NFEW \
    --batch_size "$BATCH" --seed "$SEED" $LIMIT_ARG \
    --output_path "$RESULT_DIR" $EXTRA \
  2>&1 | tee "$RESULT_DIR/run.log"
echo "[done $METHOD] $RESULT_DIR"
