#pragma once

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include "ggml.h"
#include "pidcontroller.h"
#include "rlcontroller.h"

// Async confusion matrix processing system (ACAS PID controller).
// Grasp stores per-layer alphas as float (scale_factor = 0.5*n*(alpha/10000-1)), so this
// controller mutates a float* alpha array. The contrib/precision metric is encoded into the
// I32[4] confusion payload exactly as in the SparseInfer fork, so the PID math is identical.
struct AsyncConfusionMatrix {
    int layer_id;
    int TP, FN, FP, TN;
};

class AsyncSparseController {
private:
    std::queue<AsyncConfusionMatrix> cm_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> running{true};
    std::thread worker_thread;

    BufferedPID* pids;
    int n_layers;
    float* alpha;                       // grasp: float* alphas (was int* in SparseInfer)
    const float ALPHA_MIN = 9500.0f;
    const float ALPHA_MAX = 15000.0f;
    // Controller setpoint. Default 0.995 (precision signal); ACAS_TARGET overrides it so the
    // contrib signal can drive to its own target (e.g. 0.986). Set once in the constructor.
    float TARGET_P = 0.995f;

    // ACAS_CONTROLLER=rl swaps the PID for tabular Q-learning (one RLController per layer,
    // seed = layer index). The RL push (decision-boundary units) becomes a FLOAT alpha step via
    // ALPHA_UNIT = 10000/(hidden/2), hidden=4096 (ProSparse-7B / ReLU).
    // Grasp's alpha is float*, so the push is applied as a float (no rounding), unlike the SI int*.
    bool use_rl = false;
    std::vector<RLController> rls;
    static constexpr float ALPHA_UNIT = 10000.0f / (4096.0f / 2.0f);

    void worker_loop();

public:
    AsyncSparseController(BufferedPID* pids, int n_layers, float* alpha);
    ~AsyncSparseController();

    void enqueue_payload(const AsyncConfusionMatrix& cm);
    void shutdown();

    // Statistics
    std::atomic<int> processed_count{0};
    std::atomic<int> queue_size{0};
};

// Global async controller instance
extern AsyncSparseController* g_async_controller;


// Helper functions
void init_async_controller(BufferedPID* pids, int n_layers, float* alpha);
void cleanup_async_controller();

// CUDA async transfer functions
void init_async_transfer_processing();
void cleanup_async_transfer_processing();
