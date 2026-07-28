# SparseInfer: Training-free Prediction of Activation

Fast LLM inference with low-overhead sparsity predictor and sparse kernels.

![llama](image/overview.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Abstract

Leveraging sparsity is crucial for optimizing large
language model (LLM) inference; however, modern LLMs employing SiLU as their activation function exhibit minimal activation sparsity. Recent research has proposed replacing SiLU
with ReLU to induce significant activation sparsity and showed
no downstream task accuracy degradation through fine-tuning.
However, taking full advantage of it required training a predictor
to estimate this sparsity.

We introduce SparseInfer,
a simple, light-weight, and training-free predictor for activation
sparsity of ReLU-fied LLMs, in which activation sparsity is
predicted by comparing only the sign bits of inputs and weights. To
compensate for possible prediction inaccuracy, an adaptive tuning
of the predictor’s conservativeness is enabled, which can also serve
as a control knob for optimizing LLM inference. The proposed
method achieves approximately 21% faster inference speed over
the state-of-the-art, with negligible accuracy loss of within 1%p

## TODO

-   [ ] Observing Precision and Recall based on the Alpha value.
-   [ ] Compare PowerInfer with Precision and Recall.
-   [ ] Create a 2-bit predictor.
-   [ ] Apply adaptive alpha values per layer to improve accuracy: Sort by (number of positives - number of negatives), and predict 0 for a certain proportion.

## Installation

### Download

```bash
git clone https://github.com/simpack0513/SparseInfer.git
cd SparseInfer
```

### Build

```bash
mkdir build
cd build
cmake .. -DLLAMA_CUDA=ON -DLLAMA_CUDA_MMV_Y=16
cmake --build . --config Release -j 10
```

-j : Number of CPUs for build
