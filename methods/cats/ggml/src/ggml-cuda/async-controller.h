#pragma once

#include "pid-controller.h"
#include "rlcontroller.h"
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

// Quality payload for one layer.
struct AsyncQualityMetrics {
    int layer_id;
    float error_metric;    // Error metric for threshold adjustment
    float total_metric;    // Total metric for normalization
};

class AsyncThresholdController {
private:
    std::queue<AsyncQualityMetrics> metrics_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> running{true};
    std::thread worker_thread;
    
    std::vector<BufferedPID> activation_pids;  // Only activation PIDs needed
    int n_layers;

    float TARGET_QUALITY_ACTIVATION = 0.99f;

    // ACAS_CONTROLLER=rl selects tabular Q-learning; the push scales as 1% of the threshold.
    bool use_rl = false;
    std::vector<RLController> rls;

    void worker_loop();
    
public:
    AsyncThresholdController(int n_layers, float target_quality_activation = 0.99f);
    ~AsyncThresholdController();
    
    void enqueue_payload(const AsyncQualityMetrics& metrics);
    void shutdown();
    
    std::atomic<int> processed_count{0};
    std::atomic<int> queue_size{0};
};

extern AsyncThresholdController* g_threshold_controller;

void init_async_controller(int n_layers, float target_quality_activation = 0.99f);
void cleanup_async_controller();