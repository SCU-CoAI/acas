// cadence.h — per-layer MIMD adaptive check-interval (Algorithm 1)
// (accuracy/code/modeling_*_mimd.py; MIMD:
// multiplicative increase AND multiplicative decrease). A drop-in alternative to the fixed online-
// check sampling: when ACAS_CADENCE=mimd, each layer keeps its OWN check interval that
// multiplicatively shrinks when the layer is drifting (abs_error > ERROR_THRESHOLD) and grows when it
// is stable, bounded to [MIN_INTERVAL, MAX_INTERVAL]. Default (fixed) cadence is untouched.
//
//   should_check(layer)  -> true when the layer is due for an online quality check this decode token
//                           (call once per layer per token at graph-build time; it advances the
//                            per-layer token counter and resets it when a check fires).
//   adapt(layer, |error|) -> adapt that layer's interval on the SAME error the controller optimizes
//                           (call from the controller after it computes the metric).
//
// Selected at controller init by ACAS_CADENCE=mimd (else fixed). The global g_mimd is non-null only
// in mimd mode, so a simple `if (g_mimd)` switches the gate. Per-layer state; the small main/worker
// thread race on interval_ is benign (cadence is a heuristic, not a correctness invariant).
#pragma once
#include <vector>
#include <algorithm>

class MIMDCadence {
public:
    // Multiplicative-increase/decrease check cadence (Algorithm 1). The bounds
    // [min_interval, max_interval] = [Nmin, Nmax] are ctor args so a run can override them (e.g. via
    // ACAS_NMIN/ACAS_NMAX in the controller TU) WITHOUT recompiling; defaults reproduce the original
    static constexpr float INCREASE        = 1.5f;   // grow interval when stable
    static constexpr float DECREASE        = 0.5f;   // shrink interval when drifting
    static constexpr float INIT_INTERVAL   = 200.0f;
    static constexpr float ERROR_THRESHOLD = 0.01f;

    explicit MIMDCadence(int n_layers, float min_interval = 50.0f, float max_interval = 500.0f)
        : min_interval_(min_interval), max_interval_(max_interval),
          interval_(n_layers > 0 ? n_layers : 0, INIT_INTERVAL),
          tokens_since_(n_layers > 0 ? n_layers : 0, 0) {}

    // per layer, per decode token: due for a check?
    bool should_check(int layer) {
        if (layer < 0 || layer >= (int) interval_.size()) return false;
        if (++tokens_since_[layer] >= (int) interval_[layer]) {
            tokens_since_[layer] = 0;
            return true;
        }
        return false;
    }

    // adapt this layer's interval on the controller's |metric - target|
    void adapt(int layer, float abs_error) {
        if (layer < 0 || layer >= (int) interval_.size()) return;
        if (abs_error > ERROR_THRESHOLD)
            interval_[layer] = std::max(min_interval_, interval_[layer] * DECREASE);
        else
            interval_[layer] = std::min(max_interval_, interval_[layer] * INCREASE);
    }

    float interval(int layer) const {
        return (layer >= 0 && layer < (int) interval_.size()) ? interval_[layer] : 0.0f;
    }

private:
    float              min_interval_;
    float              max_interval_;
    std::vector<float> interval_;
    std::vector<int>   tokens_since_;
};

// Global cadence manager: non-null only when ACAS_CADENCE=mimd. Created at controller init
// (n_layers known there), referenced by the build-time gate. Definition lives in the fork's
// controller TU (e.g. sparse_async.cpp / async-controller.cpp).
extern MIMDCadence * g_mimd;
