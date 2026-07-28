# ACAS-CATS (cats)

- **Method**: CATS — SiLU activation thresholding (single activation knob).
- **Model**: LLaMA-3.1-8B-Instruct, `llama-3.1-8b-instruct.gguf` (fetch separately).
- **Layout**: modular llama.cpp (`src/llama-graph.cpp`, `ggml/src/ggml-cuda/`, `tools/main/`).
- **Binary**: `build/bin/llama-cli`. Build: `cmake -B build -DGGML_CUDA=ON && cmake --build build -j`.

## ACAS knobs (this fork)
`ACAS_SIGNAL` (l2|contrib) · `ACAS_CONTROLLER` (pid|rl) · `ACAS_CADENCE` (fixed|mimd) · `ACAS_TARGET` ·
`ACAS_DENSE_UPDATE_PERCENT` · `ACAS_CONFIG` (per-layer threshold file) · `ACAS_LOG`. No gate-prediction knob (`ACAS_GATE_TARGET` n/a).

Threshold configs live in `thresholds_configs/`; pass one via `ACAS_CONFIG=<threshold file> (operating points: ../../efficiency/configs/cats/)`.
Shared `cadence.h` / `rlcontroller.h` are under `ggml/src/ggml-cuda/`.
See [INTERFACE.md](../../docs/INTERFACE.md) and [REPRODUCE.md](../../docs/REPRODUCE.md).
