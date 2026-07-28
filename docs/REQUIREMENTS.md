# Requirements

The artifact has two independent pipelines with different hardware needs. They can be
evaluated separately; neither depends on the other's outputs.

## 1. Accuracy pipeline (`accuracy/`) — any CUDA Linux box

Reproduces the GSM8K / BBH / MBPP columns of Tables 1–2, and the interval ablation (Table 5).

**Hardware**
- One NVIDIA GPU with ≥ 24 GB VRAM (both models are ≤ 16 GB fp16). We used
  RTX 4090/5090 cloud instances; any recent CUDA GPU works.
- Runs are `batch_size 1` (the controller updates per forward pass).
- Disk: ~40 GB (both models in fp16 + HF caches).

**Software (pinned by the setup scripts — no manual installs needed)**
- Linux, Python 3.10–3.12 with the venv module (Debian/Ubuntu: `apt install python3.X-venv`);
  CUDA-enabled PyTorch.
- SiLU methods (cats, catsgp): `transformers==4.54.0`, `lm_eval==0.4.9`.
- ReLU methods (si, grasp): `transformers==4.44.2`, plus the **bundled patched
  lm-evaluation-harness** (`accuracy/lm-evaluation-harness/`, installed editable by the
  setup script; patched to add the BOS token ProSparse's tokenizer omits and to import
  cleanly under transformers 4.44.2).
- `accelerate datasets sentencepiece protobuf huggingface_hub` (installed by setup).
- Explicit pin lists: `accuracy/requirements-relu.txt`, `accuracy/requirements-silu.txt`
  (the setup scripts install exactly these into isolated per-method venvs).

**Models (downloaded by the setup scripts, not redistributed here)**
- `SparseLLM/prosparse-llama-2-7b` (public).
- `meta-llama/Llama-3.1-8B-Instruct` (gated — requires an HF token with accepted access).

## 2. Efficiency pipeline (`efficiency/` + `methods/`) — NVIDIA Jetson AGX Orin

Reproduces the TPOT / Power / Std. Power / Energy columns of Tables 1–2 and the
per-token latency tables (Tables 3–4).

**Hardware**
- NVIDIA Jetson AGX Orin (64 GB devkit). Power is sampled from the on-board
  INA3221 rails (`/sys/bus/i2c/devices/1-0040/hwmon/`), so the power/energy columns are
  Jetson-specific. TPOT / per-token latency reproduce on any CUDA machine, with different
  absolute values.
- Disk: ~50 GB (two fp16 GGUF models + five CUDA builds).

**Software**
- JetPack 6.x (L4T, CUDA 12.x), CMake ≥ 3.22, gcc, Python 3.
- Models as GGUF (fp16): converted from the HF checkpoints above with
  `methods/dense-baseline/convert_hf_to_gguf.py` (see [INSTALL.md](INSTALL.md)).
