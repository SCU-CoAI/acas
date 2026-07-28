#include "async-controller.h"
#include "llama-thresholds.h"
#include "cadence.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// Global controller instance
AsyncThresholdController* g_threshold_controller = nullptr;

// ACAS_CADENCE=mimd -> per-layer adaptive check-interval (MIMD). Non-null only in mimd mode; the
// build-time gate in llama-graph.cpp consults it instead of the fixed per-mille sampling. Created in
// the controller ctor (n_layers known), torn down in cleanup_async_controller.
MIMDCadence * g_mimd = nullptr;

AsyncThresholdController::AsyncThresholdController(int n_layers, float target_quality_activation)
    : n_layers(n_layers), TARGET_QUALITY_ACTIVATION(target_quality_activation) {

    // ACAS contrib: allow the controller setpoint to be overridden (the contrib signal targets a
    // different operating point than the L2 precision quality). The quality the worker reads equals
    // the active signal (L2 precision OR contrib), so a single target knob serves both.
    if (const char * t = std::getenv("ACAS_TARGET")) {
        const float v = std::atof(t);
        if (v > 0.0f && v < 1.0f) {
            TARGET_QUALITY_ACTIVATION = v;
        }
    }

    // Initialize PID controllers for activation thresholds only
    for (int i = 0; i < n_layers; i++) {
        // PID parameters: kp=0.5, ki=0.1, kd=0.05, buffer_size=5, max_age=5
        activation_pids.emplace_back(0.5f, 0.1f, 0.05f, 5, 5);
    }

    // ACAS_CONTROLLER=rl -> tabular Q-learning per layer (seed = layer index, target =
    // TARGET_QUALITY_ACTIVATION after any ACAS_TARGET override above); default 'pid' keeps the
    // BufferedPID path byte-identical (use_rl stays false).
    // SiLU fork: pass the SiLU action set {-5,-0.7,-0.1,0,0.1,0.7,5} (matches modeling_acas_cats*.py).
    if (const char * c = std::getenv("ACAS_CONTROLLER")) {
        if (std::strcmp(c, "rl") == 0) {
            use_rl = true;
            rls.reserve(n_layers);
            for (int i = 0; i < n_layers; i++) rls.emplace_back(TARGET_QUALITY_ACTIVATION, i, RLController::silu_actions());
            const char *log = std::getenv("ACAS_LOG");
            if (log && std::atoi(log) != 0) {
                fprintf(stderr, "DEBUG: ACAS_CONTROLLER=rl -> %d RLControllers (target=%.4f)\n",
                        n_layers, TARGET_QUALITY_ACTIVATION);
            }
        }
    }

    // ACAS_CADENCE=mimd -> per-layer adaptive MIMD check-interval (replaces the fixed per-mille
    // online-check sampling). Default 'fixed' leaves g_mimd null and the existing gate untouched.
    if (const char * c = std::getenv("ACAS_CADENCE")) {
        if (std::strcmp(c, "mimd") == 0 && !g_mimd) {
            const char * en = std::getenv("ACAS_NMIN"); const char * ex = std::getenv("ACAS_NMAX");
            float nmin = en ? std::atof(en) : 50.0f;   // CATS default Nmin=50; override via ACAS_NMIN
            float nmax = ex ? std::atof(ex) : 500.0f;  // default Nmax=500; override via ACAS_NMAX
            g_mimd = new MIMDCadence(n_layers, nmin, nmax);
        }
    }

    // Start background worker thread for async processing
    worker_thread = std::thread(&AsyncThresholdController::worker_loop, this);
}

AsyncThresholdController::~AsyncThresholdController() {
    shutdown();
}

void AsyncThresholdController::shutdown() {
    running = false;
    cv.notify_all();
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

void AsyncThresholdController::enqueue_payload(const AsyncQualityMetrics& metrics) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        metrics_queue.push(metrics);
        queue_size = metrics_queue.size();
    }
    cv.notify_one();
}

void AsyncThresholdController::worker_loop() {
    while (running) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv.wait(lock, [this] { return !metrics_queue.empty() || !running; });
        
        if (!running) break;
        
        // Process all available quality metrics from the queue
        while (!metrics_queue.empty()) {
            AsyncQualityMetrics metrics = metrics_queue.front();
            metrics_queue.pop();
            queue_size = metrics_queue.size();
            lock.unlock();
            
            const char *log = std::getenv("ACAS_LOG");
            if (log && std::atoi(log) != 0) {
                fprintf(stderr, "DEBUG: worker_loop processing layer %d: error=%.6f, total=%.6f\n", 
                        metrics.layer_id, metrics.error_metric, metrics.total_metric);
            }
            
            // Process activation quality metrics asynchronously
            if (metrics.total_metric > 0) {
                float quality = 1.0f - (metrics.error_metric / metrics.total_metric);

                // Read the per-layer activation threshold (same variable both controllers write).
                float current = g_sparsity_thresholds.get_activation_threshold(metrics.layer_id);

                float error_for_pid = 0.0f;     // logged; 0 in the RL path (no logit error)
                float threshold_change;

                if (use_rl) {
                    // RL (tabular Q-learning) drop-in: push (decision-boundary units) scaled to a
                    // relative threshold step. Matches the Python:
                    //   threshold_change = push * threshold * 0.010   (1% of current threshold / unit)
                    const float push = rls[metrics.layer_id].update(quality);
                    threshold_change = push * current * 0.010f;
                } else {
                    // Calculate error using logit transformation
                    error_for_pid = calculate_error(quality, TARGET_QUALITY_ACTIVATION);

                    // Update PID controller for activation thresholds
                    // Negate error for correct control direction:
                    // If quality > target: error is negative, negate to get positive adjustment to increase threshold
                    // If quality < target: error is positive, negate to get negative adjustment to decrease threshold
                    BufferedPID& pid = activation_pids[metrics.layer_id];

                    pid.add_sample(-error_for_pid);  // Negate for correct control direction
                    float adjustment = pid.get_update();

                    // Update activation threshold using relative scaling
                    // Positive adjustment = increase threshold (more sparsity)
                    // Negative adjustment = decrease threshold (less sparsity)
                    threshold_change = adjustment * current * 0.02f;
                }

                // Same write + clamp for both controllers (matches the existing PID path / Python floor).
                float new_threshold = std::max(1e-6f, current + threshold_change);
                g_sparsity_thresholds.set_activation_threshold(metrics.layer_id, new_threshold);

                // MIMD: adapt this layer's check cadence on the same |quality - target| the controller
                // optimizes (quality == contrib in contrib mode). No-op unless ACAS_CADENCE=mimd.
                if (g_mimd) g_mimd->adapt(metrics.layer_id, std::fabs(quality - TARGET_QUALITY_ACTIVATION));

                const char *log = std::getenv("ACAS_LOG");
                if (log && std::atoi(log) != 0) {
                    fprintf(stderr, "Layer %d Activation: quality=%.4f, error=%.4f, threshold=%.4f->%.4f\n",
                            metrics.layer_id, quality, error_for_pid, current, new_threshold);
                }

                processed_count++;
            }
            
            lock.lock();
        }
    }
}

// Helper functions for controller lifecycle
void init_async_controller(int n_layers, float target_quality_activation) {
    if (g_threshold_controller) {
        delete g_threshold_controller;
    }
    g_threshold_controller = new AsyncThresholdController(n_layers, target_quality_activation);
}

void cleanup_async_controller() {
    if (g_threshold_controller) {
        delete g_threshold_controller;
        g_threshold_controller = nullptr;
    }
    if (g_mimd) {
        delete g_mimd;
        g_mimd = nullptr;
    }
}
