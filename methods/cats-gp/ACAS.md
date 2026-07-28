# ACAS-CATS-GP (cats-gp)

- **Method**: CATS + Grasp group-based predictor — SiLU, **two knobs** (activation + predictor).
- **Model**: LLaMA-3.1-8B-Instruct, `llama-3.1-8b-instruct.gguf` (fetch separately).
- **Layout**: modular llama.cpp (`src/llama-graph.cpp`, `ggml/src/ggml-cuda/`, `tools/main/`).
- **Binary**: `build/bin/llama-cli`. Build: `cmake -B build -DGGML_CUDA=ON && cmake --build build -j`.

## ACAS knobs (this fork)
`ACAS_SIGNAL` (l2|contrib) · `ACAS_CONTROLLER` (pid|rl) · `ACAS_CADENCE` (fixed|mimd) · `ACAS_TARGET`
(activation) · **`ACAS_GATE_TARGET`** (predictor — this is the only fork that uses it) ·
`ACAS_DENSE_UPDATE_PERCENT` · `ACAS_CONFIG` · `ACAS_LOG`.

Two threshold knobs: activation (`ACAS_TARGET`) + predictor (`ACAS_GATE_TARGET`, signal-routed). In
contrib mode the cadence adapts on `max(activation_err, predictor_err)`. Threshold configs live in
`thresholds_configs/grasp+act/`; pass via `ACAS_CONFIG=`. Shared `cadence.h` / `rlcontroller.h` are
under `ggml/src/ggml-cuda/`. See [INTERFACE.md](../../docs/INTERFACE.md) and [REPRODUCE.md](../../docs/REPRODUCE.md).
