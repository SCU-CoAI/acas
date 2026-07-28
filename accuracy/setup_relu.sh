#!/bin/bash
# =====================================================================================
# SETUP for the ReLU methods (SI, Grasp on ProSparse-LLaMA-2-7B).
# Creates a SEPARATE venv per method (both transformers 4.44.2) AND a per-method model dir
# whose large weights are SYMLINKED to one shared download (so si and grasp each own their
# modeling_sparsellama.py and can run in parallel on different GPUs without colliding). Run ONCE.
#
#   bash setup_relu.sh                 # both methods
#   ONLY=si bash setup_relu.sh         # just one (single-GPU instance)
# (ProSparse is public; export HF_TOKEN only if you hit rate limits.)
# =====================================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS="${RELU_WEIGHTS_DIR:-$HOME/prosparse-weights}"   # the one shared model download
METHODS="${ONLY:-si grasp}"

pin() {  # $1 = venv path
  "${PYTHON:-python3}" -m venv "$1"
  "$1/bin/pip" install -q --upgrade pip
  "$1/bin/python" -c "import torch" 2>/dev/null || "$1/bin/pip" install -q torch
  "$1/bin/pip" install -q "transformers==4.44.2" "lm_eval==0.4.9" accelerate datasets sentencepiece protobuf huggingface_hub
  # The bundled lm-evaluation-harness must replace the PyPI install: it patches
  # models/huggingface.py to prepend the BOS token that ProSparse-LLaMA-2's tokenizer omits
  # on batch encode (generation tasks score wrong without it), and fixes the Qwen2Audio
  # eager import that crashes on transformers 4.44.2.
  "$1/bin/pip" install -q -e "$HERE/lm-evaluation-harness"
}

build_dir() {  # $1 = per-method model dir (symlink weights, copy the .py files)
  mkdir -p "$1"
  for f in "$WEIGHTS"/*; do b="$(basename "$f")"; case "$b" in
    modeling_sparsellama.py|configuration_sparsellama.py) cp -f "$f" "$1/$b";;
    *) ln -sf "$f" "$1/$b";; esac; done
}

FIRST_VENV=""
for m in $METHODS; do
  case "$m" in si|grasp) ;; *) echo "skip unknown $m"; continue;; esac
  VENV="$HOME/acas_venv_$m"
  echo "=== [$m] venv ($VENV) + transformers 4.44.2 ==="
  pin "$VENV"; [ -z "$FIRST_VENV" ] && FIRST_VENV="$VENV"
done

if [ ! -f "$WEIGHTS/modeling_sparsellama.py" ]; then
  echo "=== download SparseLLM/prosparse-llama-2-7b -> $WEIGHTS (once) ==="
  HF_TOKEN="${HF_TOKEN:-}" "$FIRST_VENV/bin/huggingface-cli" download SparseLLM/prosparse-llama-2-7b --local-dir "$WEIGHTS"
fi

for m in $METHODS; do
  case "$m" in
    si)    SRC="modeling_acas_si_mimd.py";;
    grasp) SRC="modeling_acas_grasp_mimd.py";;
    *) continue;; esac
  VENV="$HOME/acas_venv_$m"; MDIR="$HOME/prosparse-$m"
  echo "=== [$m] build model dir ($MDIR, weights symlinked) ==="
  build_dir "$MDIR"
  MODEL_FILE="$MDIR/modeling_sparsellama.py"
  [ -f "$MODEL_FILE.stock_bak" ] || cp "$MODEL_FILE" "$MODEL_FILE.stock_bak"
  # prime + back up the per-method HF remote-code cache (distinct dir name => no cross-method clash)
  "$VENV/bin/python" - "$MDIR" <<'PY' || true
import sys
from transformers import AutoConfig
AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
print("remote-code cache primed")
PY
  CACHE_FILE="$HOME/.cache/huggingface/modules/transformers_modules/prosparse-$m/modeling_sparsellama.py"
  if [ -f "$CACHE_FILE" ]; then [ -f "$CACHE_FILE.stock_bak" ] || cp "$CACHE_FILE" "$CACHE_FILE.stock_bak"; else CACHE_FILE=""; fi
  cat > "$HERE/.method_$m.env" <<EOF
VENV="$VENV"
MODEL="$MDIR"
DEPLOY="$MODEL_FILE"
DEPLOY_CACHE="$CACHE_FILE"
TRUST=",trust_remote_code=True"
EOF
  "$VENV/bin/python" -m py_compile "$HERE/code/$SRC" && echo "  [$m] compile OK -> .method_$m.env" || echo "  [$m] COMPILE FAIL"
done
echo "=== ReLU SETUP DONE. methods: $METHODS  shared weights: $WEIGHTS ==="
echo "Run e.g.:  bash run_bench.sh --method si --target 0.986 --task bbh --gpu 2"
