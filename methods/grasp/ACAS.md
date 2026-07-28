# ACAS-GR (grasp)

- **Method**: Grasp — ReLU sparsity with a group/outlier predictor (built on the SparseInfer base).
- **Model**: ProSparse-LLaMA-2-7B (ReLU-fied), `prosparse-llama-2-7b.gguf` (fetch separately).
- **Layout**: monolithic llama.cpp (`llama.cpp`, `ggml-cuda.cu`, `examples/main/`).
- **Binary**: `build/bin/main`. Build: `cmake -B build -DLLAMA_CUDA=ON && cmake --build build -j`.
- **Fork of**: [Sogang-aisys/Grasp](https://github.com/Sogang-aisys/Grasp).

## ACAS knobs (this fork)
`ACAS_SIGNAL` (precision|contrib) · `ACAS_CONTROLLER` (pid|rl) · `ACAS_CADENCE` (fixed|mimd) ·
`ACAS_TARGET` · `ACAS_CONFIG` (initial per-layer alphas, one per line) ·
`ACAS_DENSE_UPDATE_PERCENT` · `ACAS_LOG`. No gate-prediction knob (`ACAS_GATE_TARGET` n/a).

The default (no `ACAS_SIGNAL=contrib`) path is the original static predictor sparse path;
`ACAS_SIGNAL=contrib` enables the online contrib controller (float per-layer alphas).
Shared `cadence.h` / `rlcontroller.h` are at the fork root.
See [INTERFACE.md](../../docs/INTERFACE.md) and [REPRODUCE.md](../../docs/REPRODUCE.md).
