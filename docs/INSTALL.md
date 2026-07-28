# Installation

Two independent pipelines. Install what you need (see [REQUIREMENTS.md](REQUIREMENTS.md)).

## Efficiency pipeline (Jetson AGX Orin)

### Build the five methods

```bash
export PATH=/usr/local/cuda/bin:$PATH   # ensure nvcc is visible (non-login shells)
for m in sparseinfer grasp; do          # monolithic llama.cpp generation
  cd methods/$m && cmake -B build -DLLAMA_CUDA=ON && cmake --build build -j --target main && cd ../..
done
for m in cats cats-gp dense-baseline; do  # modular generation
  cd methods/$m && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF && cmake --build build -j --target llama-cli && cd ../..
done
```

Binaries land at `methods/{sparseinfer,grasp}/build/bin/main` (ReLU, monolithic llama.cpp) and
`methods/{cats,cats-gp,dense-baseline}/build/bin/llama-cli` (SiLU, modular llama.cpp).

### Models

Download the HF checkpoints and convert once (conversion script located in the
dense-baseline dir):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download SparseLLM/prosparse-llama-2-7b     --local-dir ~/prosparse-hf
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct   --local-dir ~/llama31-hf \
  --exclude "original/*"          # gated: needs HF_TOKEN with accepted access

python3 methods/dense-baseline/convert_hf_to_gguf.py ~/prosparse-hf \
  --outtype f16 --outfile ~/prosparse-llama-2-7b.gguf
python3 methods/dense-baseline/convert_hf_to_gguf.py ~/llama31-hf \
  --outtype f16 --outfile ~/llama-3.1-8b-instruct.gguf
```

The efficiency scripts default to `~/prosparse-llama-2-7b.gguf` and
`~/llama-3.1-8b-instruct.gguf`; override with `PROSPARSE=` / `LLAMA=` env vars.

### Basic usage check

All four forks take their per-layer config file (initial alphas for the ReLU forks,
thresholds for the SiLU forks) via `ACAS_CONFIG=`; the **ReLU forks** (sparseinfer, grasp)
use the `main` binary, the **SiLU forks** (cats, cats-gp) use `llama-cli`.

ReLU (SparseInfer shown; grasp is identical with its own config):

```bash
cd methods/sparseinfer
ACAS_SIGNAL=contrib ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.986 \
ACAS_CONFIG=../../efficiency/configs/sparseinfer/contrib_0.986.txt \
  build/bin/main -m ~/prosparse-llama-2-7b.gguf -ngl 33 \
  -p "Once upon a time, " -n 24
```

SiLU (CATS shown; cats-gp additionally takes `ACAS_GATE_TARGET=`):

```bash
cd methods/cats
ACAS_SIGNAL=contrib ACAS_CONTROLLER=pid ACAS_CADENCE=fixed ACAS_TARGET=0.89 \
ACAS_CONFIG=../../efficiency/configs/cats/contrib_0.89.txt \
  build/bin/llama-cli -m ~/llama-3.1-8b-instruct.gguf -ngl 33 \
  -p "Once upon a time, " -n 24 -no-cnv
```

Swap `ACAS_CONTROLLER=rl` / `ACAS_CADENCE=mimd` to smoke-test the other toggles
([INTERFACE.md](INTERFACE.md) documents all knobs and the per-method customizations).

## Accuracy pipeline (any CUDA Linux box)

From the repo root:

```bash
cd accuracy
# SiLU methods (cats, catsgp): needs HF_TOKEN for the gated Llama-3.1 download
export HF_TOKEN=hf_xxx
bash setup_silu.sh          # per-method venvs + model download (once per box)
# ReLU methods (si, grasp): ProSparse is public, no token needed
bash setup_relu.sh
```

Setup creates isolated venvs (`~/acas_venv_<method>`) with pinned versions, downloads the
models, backs up the stock modeling files, and writes `.method_<m>.env` descriptors that
`run_bench.sh` consumes. `ONLY=si bash setup_relu.sh` sets up a single method.

### Smoke run

```bash
bash run_bench.sh --method si --task gsm8k --limit 3
```

Expected: `[deploy si] contrib …`, an lm-eval progress bar, a results JSON under
`accuracy/runs/`, and `[restore si] stock modeling files` on exit.
