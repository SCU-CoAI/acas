<div align="center">

# ACAS

[![Paper](http://img.shields.io/badge/paper-CODES-B31B1B.svg)](docs/CODES_ACAS_Camera_Ready.pdf)
[![Model](https://img.shields.io/badge/model-ProSparse--LLaMA--2--7B-yellow)](https://huggingface.co/SparseLLM/prosparse-llama-2-7b)
[![Model](https://img.shields.io/badge/model-Llama--3.1--8B--Instruct-yellow)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![llama.cpp](https://img.shields.io/badge/contains-llama.cpp%20forks-blue)](https://github.com/ggml-org/llama.cpp)
[![SparseInfer](https://img.shields.io/badge/contains-SparseInfer%20fork-blue)](https://github.com/Sogang-aisys/Sparseinfer)
[![Grasp](https://img.shields.io/badge/contains-Grasp%20fork-blue)](https://github.com/Sogang-aisys/Grasp)

</div>

## Description

Artifact for the paper _"ACAS: Adaptive Control of Inference-Time Activation Sparsity for
On-Device LLMs"_ (CODES 2026). Two independent pipelines reproduce the paper's
experiments:

| Pipeline | Hardware | Reproduces |
|---|---|---|
| [`accuracy/`](accuracy/) | any CUDA Linux box | GSM8K / BBH / MBPP columns (Tables 1–2), interval ablation (Table 5) |
| [`efficiency/`](efficiency/) + [`methods/`](methods/) | Jetson AGX Orin | TPOT / Power / Energy columns (Tables 1–2), per-token latency (Tables 3–4) |

Each `methods/*` tree is fully self-contained: it carries its own copies of the ACAS components.

Artifact documentation: [INSTALL.md](docs/INSTALL.md) · [REQUIREMENTS.md](docs/REQUIREMENTS.md) ·
[STATUS.md](docs/STATUS.md) · [REPRODUCE.md](docs/REPRODUCE.md) ·
[INTERFACE.md](docs/INTERFACE.md).

## Installation

### Efficiency pipeline (Jetson AGX Orin)

```bash
git clone https://github.com/SCU-CoAI/acas.git && cd acas
export PATH=/usr/local/cuda/bin:$PATH
for m in sparseinfer grasp; do          # monolithic llama.cpp generation
  cd methods/$m && cmake -B build -DLLAMA_CUDA=ON && cmake --build build -j --target main && cd ../..
done
for m in cats cats-gp dense-baseline; do  # modular generation
  cd methods/$m && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF && cmake --build build -j --target llama-cli && cd ../..
done
```

### Accuracy pipeline (any CUDA Linux box)

```bash
cd accuracy
export HF_TOKEN=hf_xxx        # only for the gated Llama-3.1 download (SiLU methods)
bash setup_relu.sh            # SI + Grasp: pinned venvs + public ProSparse download
bash setup_silu.sh            # CATS + CATS-GP: pinned venvs + Llama-3.1 download
```

Python 3.10–3.12 (`PYTHON=python3.10 bash setup_relu.sh` if your default is newer).

## Setting up

### Models (GGUF, efficiency pipeline only)

Convert the HF checkpoints once with the bundled converter (see [INSTALL.md](docs/INSTALL.md)):

```bash
python3 methods/dense-baseline/convert_hf_to_gguf.py ~/prosparse-hf --outtype f16 --outfile ~/prosparse-llama-2-7b.gguf
python3 methods/dense-baseline/convert_hf_to_gguf.py ~/llama31-hf  --outtype f16 --outfile ~/llama-3.1-8b-instruct.gguf
```

The accuracy pipeline downloads its models automatically during setup.

## How to run

1. **Smoke test**: one accuracy row at `--limit 3`:

    ```bash
    bash accuracy/run_bench.sh --method si --task gsm8k --limit 3
    ```

2. **Any table row** — the three toggles map 1:1 to the tables
   (**Signal** = `--signal`, **Ctrl** = `--controller`, `*` = `--cadence mimd`):

    ```bash
    bash accuracy/run_bench.sh --method {si|grasp|cats|catsgp} \
        --signal {precision|activation|baseline|contrib} \
        --controller {pid|rl} --cadence {fixed|mimd} --task {gsm8k|bbh|mbpp}
    ```

3. **Full efficiency sweep**:

    ```bash
    mkdir -p efficiency/results
    tmux new -d -s sweep 'bash efficiency/sweep_table.sh > efficiency/results/sweep.log 2>&1'
    tmux new -d -s base  'bash efficiency/baselines.sh  > efficiency/results/baselines.log 2>&1'
    ```

4. **Per-token latency + interval ablation**: `efficiency/per_token_latency/` and
   `accuracy/interval_ablation/`.

Full details in [REPRODUCE.md](docs/REPRODUCE.md).
