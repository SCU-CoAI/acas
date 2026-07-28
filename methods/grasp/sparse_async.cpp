#include "sparse_async.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <ctime>
#include "common/log.h"
#include "cadence.h"

// Global controller instance
AsyncSparseController* g_async_controller = nullptr;

// ACAS_CADENCE=mimd -> per-layer adaptive check-interval (MIMD). Non-null only in mimd mode; the
// build-time gate in llama.cpp consults it instead of the fixed per-mille sampling. Created in the
// controller ctor (n_layers known), torn down in cleanup_async_controller.
MIMDCadence * g_mimd = nullptr;

AsyncSparseController::AsyncSparseController(BufferedPID* pids, int n_layers, float* alpha)
    : pids(pids), n_layers(n_layers), alpha(alpha) {

    // Allow the controller setpoint to be overridden (e.g. contrib signal targets ~0.986 instead
    // of the precision default 0.995). The encoded "precision" the worker reads equals the chosen
    // quality signal (precision OR contrib), so a single target knob serves both.
    if (const char * t = getenv("ACAS_TARGET")) {
        const float v = atof(t);
        if (v > 0.0f && v < 1.0f) {
            TARGET_P = v;
        }
    }

    // ACAS_CONTROLLER=rl -> tabular Q-learning per layer (seed = layer index, target = TARGET_P);
    // default (unset/anything else) keeps the BufferedPID path byte-identical.
    if (const char * c = getenv("ACAS_CONTROLLER")) {
        if (std::strcmp(c, "rl") == 0) {
            use_rl = true;
            rls.reserve(n_layers);
            for (int i = 0; i < n_layers; ++i) rls.emplace_back(TARGET_P, i);
        }
    }

    // ACAS_CADENCE=mimd -> per-layer adaptive MIMD check-interval (replaces the fixed per-mille
    // online-check sampling). Default 'fixed' leaves g_mimd null and the existing gate untouched.
    if (const char * c = getenv("ACAS_CADENCE")) {
        if (std::strcmp(c, "mimd") == 0 && !g_mimd) {
            const char * en = getenv("ACAS_NMIN"); const char * ex = getenv("ACAS_NMAX");
            float nmin = en ? atof(en) : 50.0f;   // Grasp default Nmin=50; override via ACAS_NMIN
            float nmax = ex ? atof(ex) : 500.0f;  // default Nmax=500; override via ACAS_NMAX
            g_mimd = new MIMDCadence(n_layers, nmin, nmax);
        }
    }

    // Start worker thread (logging will use main log file)
    worker_thread = std::thread(&AsyncSparseController::worker_loop, this);
}

AsyncSparseController::~AsyncSparseController() {
    shutdown();
}

void AsyncSparseController::shutdown() {
    running = false;
    cv.notify_all();
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

void AsyncSparseController::enqueue_payload(const AsyncConfusionMatrix& cm) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        cm_queue.push(cm);
        queue_size = cm_queue.size();
    }
    cv.notify_one();
}

void AsyncSparseController::worker_loop() {
    while (running) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv.wait(lock, [this] { return !cm_queue.empty() || !running; });

        if (!running) break;

        // Process all available confusion matrices
        while (!cm_queue.empty()) {
            AsyncConfusionMatrix cm = cm_queue.front();
            cm_queue.pop();
            queue_size = cm_queue.size();
            lock.unlock();

            // Process confusion matrix (CPU work while GPU continues).
            // For the contrib signal precision == contrib (TP=contrib*S, FP=S-TP), so the PID is
            // driven directly by the output-contribution metric without any controller change.
            const int total = cm.TP + cm.FP + cm.TN + cm.FN;
            if (total > 0) {
                const float tp_fp = cm.TP + cm.FP;
                float precision = (tp_fp > 0) ? (cm.TP / tp_fp) : TARGET_P;

                // RL (tabular Q-learning) or PID, selected once at init by ACAS_CONTROLLER.
                // Grasp keeps the alpha step as a FLOAT (no rounding); the PID path is unchanged.
                float error = 0.0f;
                float adjustment;
                if (use_rl) {
                    const float push = rls[cm.layer_id].update(precision);  // quality -> push (decision units)
                    adjustment = push * ALPHA_UNIT;                          // float alpha step, no rounding
                } else {
                    error = calculate_error(precision, TARGET_P);
                    // Update PID controller (now manages its own step counter)
                    pids[cm.layer_id].add_sample(error);
                    adjustment = pids[cm.layer_id].get_update();
                }

                // Update alpha value (grasp: float alpha mutated directly, then clamped).
                const float prev = alpha[cm.layer_id];
                alpha[cm.layer_id] += adjustment;
                alpha[cm.layer_id] = std::min(std::max(alpha[cm.layer_id], ALPHA_MIN), ALPHA_MAX);

                // MIMD: adapt this layer's check cadence on the same |metric - target| the controller
                // optimizes (precision == contrib in contrib mode). No-op unless ACAS_CADENCE=mimd.
                if (g_mimd) g_mimd->adapt(cm.layer_id, std::fabs(precision - TARGET_P));

                processed_count++;

                // Log every async update using the proper logging system.
                // precision == contrib for the contrib signal; error<0 => contrib>target => alpha
                // pushed down (more sparsity), error>0 => contrib<target => alpha pushed up.
                static const bool acas_log = []{ const char * l = getenv("ACAS_LOG"); return l && atoi(l) != 0; }();
                if (acas_log) LOG("Layer %d: precision=%.4f, error=%.4f, alpha=%.2f->%.2f\n",
                   cm.layer_id, precision, error, prev, alpha[cm.layer_id]);
            }

            lock.lock();
        }
    }
}


void init_async_controller(BufferedPID* pids, int n_layers, float* alpha) {
    if (g_async_controller) {
        delete g_async_controller;
    }
    g_async_controller = new AsyncSparseController(pids, n_layers, alpha);
}

void cleanup_async_controller() {
    if (g_async_controller) {
        delete g_async_controller;
        g_async_controller = nullptr;
    }
    if (g_mimd) {
        delete g_mimd;
        g_mimd = nullptr;
    }
}
