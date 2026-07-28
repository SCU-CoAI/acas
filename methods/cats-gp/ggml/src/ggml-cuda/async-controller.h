#pragma once

#include "pid-controller.h"
#include "rlcontroller.h"
#include <atomic>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

// Quality payload for one layer. data[] holds the transferred words; the flags select the
// consuming knob (activation, predictor audit [tp,fp,fn,tn], or predictor contrib drift).
struct AsyncQualityMetrics {
    int layer_id;
    float data[4];
    bool is_activation;
    bool is_predictor_contrib;
};

class AsyncThresholdController {
private:
    std::queue<AsyncQualityMetrics> metrics_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> running{true};
    std::thread worker_thread;

    std::vector<BufferedPID> activation_pids;
    std::vector<BufferedPID> predictor_pids;
    int n_layers;

    // ACAS_CONTROLLER=rl selects tabular Q-learning; the push scales as 1% of the threshold.
    bool use_rl = false;
    std::vector<RLController> rls;            // activation-threshold RL
    std::vector<RLController> predictor_rls;  // predictor-threshold RL (catsgr 2nd knob)

    float TARGET_QUALITY_ACTIVATION = 0.99f;
    float TARGET_QUALITY_PREDICTOR  = 0.99f;

    // Predictor-drift target for contrib mode (lower is better; default 0.05).
    float TARGET_PREDICTOR_CONTRIB  = 0.05f;

    void worker_loop();
    void apply_update(const AsyncQualityMetrics& m);

    // ACAS MIMD: most-recent activation |quality - target| per layer, cached when the activation
    // branch runs so the predictor branch can adapt the cadence on combined_err = max(act_err,
    // pred_err). Sized n_layers in the ctor; only read in mimd mode.
    std::vector<float> last_act_err_;

public:
    AsyncThresholdController(int n_layers, float target_quality_activation = 0.99f, float target_quality_predictor = 0.99f);
    ~AsyncThresholdController();

    void enqueue_payload(const AsyncQualityMetrics& metrics);
    void shutdown();

    std::atomic<int> processed_count{0};
    std::atomic<int> queue_size{0};
    
};

extern AsyncThresholdController* g_threshold_controller;

void init_async_controller(int n_layers, float target_quality_activation = 0.99f);
void cleanup_async_controller();
