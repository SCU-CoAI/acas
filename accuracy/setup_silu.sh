#!/bin/bash
# =====================================================================================
# SETUP for the SiLU methods (CATS, CATS-GP on Llama-3.1-8B-Instruct).
# Creates a SEPARATE venv per method (both transformers 4.54.0) so cats and catsgp can run
# in parallel on different GPUs of one box without colliding on modeling_llama.py. The model
# is downloaded ONCE and shared by path. Run ONCE per instance.
#
#   export HF_TOKEN=hf_xxx   (token with Llama-3.1 license accepted)
#   bash setup_silu.sh                 # both methods
#   ONLY=cats bash setup_silu.sh       # just one method (e.g. a single-GPU instance)
# =====================================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SILU_MODEL:-$HOME/Llama-3.1-8B-Instruct}"
METHODS="${ONLY:-cats catsgp}"
[ -n "${HF_TOKEN:-}" ] || { echo "ERROR: export HF_TOKEN=hf_xxx first (Llama-3.1 is gated)"; exit 1; }

# shared deps installer
pin() {  # $1 = venv path
  "${PYTHON:-python3}" -m venv "$1"
  "$1/bin/pip" install -q --upgrade pip
  "$1/bin/python" -c "import torch" 2>/dev/null || "$1/bin/pip" install -q torch
  "$1/bin/pip" install -q "transformers==4.54.0" "lm_eval==0.4.9" accelerate datasets sentencepiece protobuf huggingface_hub
}

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  echo "=== download Llama-3.1-8B-Instruct -> $MODEL_DIR (once) ==="
  # use the first venv's hub once it exists; bootstrap a tiny env if none yet
  python3 -m pip install -q --user huggingface_hub 2>/dev/null || true
  HF_TOKEN="$HF_TOKEN" python3 -m huggingface_hub.commands.huggingface_cli download \
      meta-llama/Llama-3.1-8B-Instruct --local-dir "$MODEL_DIR" --exclude "original/*" "*.pth" 2>/dev/null \
   || { echo "bootstrap hub download failed; will retry via a method venv"; }
fi

for m in $METHODS; do
  case "$m" in
    cats)   SRC="modeling_acas_cats_mimd.py";;
    catsgp) SRC="modeling_acas_catsgp_mimd.py";;
    *) echo "skip unknown method $m"; continue;; esac
  VENV="$HOME/acas_venv_$m"
  echo "=== [$m] venv ($VENV) + transformers 4.54.0 ==="
  pin "$VENV"
  # ensure model present (retry via this venv if bootstrap failed)
  [ -f "$MODEL_DIR/config.json" ] || HF_TOKEN="$HF_TOKEN" "$VENV/bin/huggingface-cli" download \
      meta-llama/Llama-3.1-8B-Instruct --local-dir "$MODEL_DIR" --exclude "original/*" "*.pth"
  LLAMA_PY="$("$VENV/bin/python" -c 'import transformers.models.llama as M,os;print(os.path.join(os.path.dirname(M.__file__),"modeling_llama.py"))')"
  [ -f "$LLAMA_PY.stock_bak" ] || cp "$LLAMA_PY" "$LLAMA_PY.stock_bak"
  cat > "$HERE/.method_$m.env" <<EOF
VENV="$VENV"
MODEL="$MODEL_DIR"
DEPLOY="$LLAMA_PY"
TRUST=""
EOF
  # sanity import
  cp "$HERE/code/$SRC" "$(dirname "$LLAMA_PY")/_acas_imptest.py"
  ACAS_Q1_TAP=0 ACAS_CATSGR_S_VALUES="$HERE/code/3std_averages_s_values.csv" \
    "$VENV/bin/python" -c "import transformers.models.llama._acas_imptest" 2>&1 \
    | grep -viE '🚨|auto_docstring' | grep -iE 'error|traceback' && echo "  [$m] IMPORT FAIL" || echo "  [$m] import OK"
  rm -f "$(dirname "$LLAMA_PY")/_acas_imptest.py"
  echo "  [$m] ready -> .method_$m.env"
done
echo "=== SiLU SETUP DONE. methods: $METHODS  model: $MODEL_DIR ==="
echo "Run e.g.:  bash run_bench.sh --method cats --target 0.89 --task bbh --gpu 0"
