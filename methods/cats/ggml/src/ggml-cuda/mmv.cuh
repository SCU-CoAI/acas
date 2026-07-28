#include "common.cuh"

void ggml_cuda_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

void ggml_cuda_op_mul_mat_vec(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmv(enum ggml_type type, int cc, const int64_t * src0_ne, int64_t ne11);

// Dense-derived matvec with per-row gating: out = gate ⊙ (A·x [+b] [×s])
void ggml_cuda_mul_mat_vec_gated(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,     // matrix A, types: F32/F16/BF16
    const ggml_tensor * src1,     // vector x, F32
    const ggml_tensor * gate,     // per-row gate, F32
    const ggml_tensor * bias,     // optional per-row bias, F32 or NULL
    const ggml_tensor * scale,    // optional per-row scale, F32 or NULL
    ggml_tensor       * dst);     // output, F32
