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
// build-time gate in llama-graph.cpp consults it instead of the fixed per-mille sampling. CATS-GP
// (2 knobs) adapts on combined_err = max(act_err, pred_err). Created in the controller ctor,
// torn down in cleanup_async_controller.
MIMDCadence * g_mimd = nullptr;

AsyncThresholdController::AsyncThresholdController(int n_layers, float target_quality_activation, float target_quality_predictor) 
    : n_layers(n_layers), TARGET_QUALITY_ACTIVATION(target_quality_activation), TARGET_QUALITY_PREDICTOR(target_quality_predictor) {
    // predictor wrong-skip target for the audit (L2) path; default 0.05 (paper operating point).
    TARGET_QUALITY_PREDICTOR = 0.05f;

    // ACAS_TARGET: single activation-knob setpoint. Serves both signals (the worker reads the active
    // L2/precision OR contrib signal as "quality"), so one target knob is enough. Default 0.99.
    if (const char * t = std::getenv("ACAS_TARGET")) {
        const float v = std::atof(t);
        if (v > 0.0f && v < 1.0f) {
            TARGET_QUALITY_ACTIVATION = v;
        }
    }

    // ACAS_SIGNAL=contrib selects the predictor knob's driving signal (drift instead of the audit
    // wrong-skip); used below both for the ACAS_GATE_TARGET routing and the RL predictor target.
    const bool contrib_signal = []{
        const char * s = std::getenv("ACAS_SIGNAL");
        return s && std::strcmp(s, "contrib") == 0;
    }();

    // ACAS_GATE_TARGET: single predictor-knob setpoint, routed to the active signal's predictor
    // target -- the contrib drift target in contrib mode (default 0.05), or the audit wrong-skip
    // target in L2 mode (default 0.05). Mirrors how ACAS_TARGET serves the activation knob for both signals.
    if (const char * t = std::getenv("ACAS_GATE_TARGET")) {
        const float v = std::atof(t);
        if (v >= 0.0f && v < 1.0f) {
            if (contrib_signal) TARGET_PREDICTOR_CONTRIB = 1.0f - v;   // quality target -> drift target
            else                TARGET_QUALITY_PREDICTOR = v;
        }
    }

    // Initialize PID controllers for activation thresholds only
    for (int i = 0; i < n_layers; i++) {
        // PID parameters: kp=0.5, ki=0.1, kd=0.05, buffer_size=5, max_age=5
        activation_pids.emplace_back(0.5f, 0.1f, 0.05f, 5, 5);
    }
    for (int i = 0; i < n_layers; i++) {
        predictor_pids.emplace_back(1.0f, 0.2f, 0.15f, 5, 5);
    }

    // ACAS_CONTROLLER=rl -> tabular Q-learning per layer on the ACTIVATION threshold knob
    // (seed = layer index, target = TARGET_QUALITY_ACTIVATION after any ACAS_TARGET override).
    // Default 'pid' keeps the BufferedPID activation path byte-identical.
    // SiLU fork: pass the SiLU action set {-5,-0.7,-0.1,0,0.1,0.7,5} (matches modeling_acas_catsgp*.py).
    if (const char * c = std::getenv("ACAS_CONTROLLER")) {
        if (std::strcmp(c, "rl") == 0) {
            use_rl = true;
            rls.reserve(n_layers);
            predictor_rls.reserve(n_layers);
            for (int i = 0; i < n_layers; i++) {
                rls.emplace_back(TARGET_QUALITY_ACTIVATION, i, RLController::silu_actions());
                // predictor knob: separate RL per layer, seed offset 10000, target =
                // the active signal's predictor target (drift target in contrib mode, wrong-skip
                // target in L2 mode) — same as Python's target_wrong_skip. Signal = drift/wrong_skip.
                predictor_rls.emplace_back(contrib_signal ? TARGET_PREDICTOR_CONTRIB : TARGET_QUALITY_PREDICTOR,
                                           10000 + i, RLController::silu_actions());
            }
        }
    }

    // ACAS_CADENCE=mimd -> per-layer adaptive MIMD check-interval (replaces the fixed per-mille
    // online-check sampling). Default 'fixed' leaves g_mimd null and the existing gate untouched.
    last_act_err_.assign(n_layers, 0.0f);
    if (const char * c = std::getenv("ACAS_CADENCE")) {
        if (std::strcmp(c, "mimd") == 0 && !g_mimd) {
            const char * en = std::getenv("ACAS_NMIN"); const char * ex = std::getenv("ACAS_NMAX");
            float nmin = en ? std::atof(en) : 50.0f;   // CATS-GP default Nmin=50; override via ACAS_NMIN
            float nmax = ex ? std::atof(ex) : 500.0f;  // default Nmax=500; override via ACAS_NMAX
            g_mimd = new MIMDCadence(n_layers, nmin, nmax);
        }
    }

    worker_thread = std::thread(&AsyncThresholdController::worker_loop, this);
}



void AsyncThresholdController::apply_update(const AsyncQualityMetrics & m) {
    const float * data = m.data;
    const int layer_id = m.layer_id;
    const bool is_activation = m.is_activation;
    const bool is_predictor_contrib = m.is_predictor_contrib;
    if (is_activation) {
        // Process activation threshold updates directly
        if (data[1] > 0) {  // total_metric > 0
            float quality = 1.0f - (data[0] / data[1]);  // 1 - (error/total)

            // RL (tabular Q-learning) or PID, selected once at init by ACAS_CONTROLLER.
            // Both drive the SAME per-layer activation threshold + clamp; only the step rule differs.
            float current = g_sparsity_thresholds.get_activation_threshold(layer_id);
            float error_for_pid = 0.0f;
            float adjustment = 0.0f;
            float threshold_change;
            if (use_rl) {
                // SiLU apply scale: 1% of current threshold per push-unit (validated RL apply scale).
                const float push = rls[layer_id].update(quality);  // quality -> push (decision units)
                threshold_change = push * current * 0.010f;
            } else {
                error_for_pid = calculate_error(quality, TARGET_QUALITY_ACTIVATION);
                // Update PID controller for activation thresholds
                BufferedPID& pid = activation_pids[layer_id];
                pid.add_sample(-error_for_pid);  // Negate for correct control direction
                adjustment = pid.get_update();
                threshold_change = adjustment * current * 0.02f;  // Fixed step size
            }
            float new_threshold = std::max(1e-6f, current + threshold_change);
            g_sparsity_thresholds.set_activation_threshold(layer_id, new_threshold);

            // MIMD: cache this layer's activation error so the predictor branch (which runs second on
            // a check token) can adapt the cadence on combined_err = max(act_err, pred_err).
            if (g_mimd) last_act_err_[layer_id] = std::fabs(quality - TARGET_QUALITY_ACTIVATION);

            const char *log = std::getenv("ACAS_LOG");
            if (log && std::atoi(log) != 0) {
                fprintf(stderr, "DEBUG: apply_update (activation) layer %d: error=%.6f, total=%.6f\n",
                        layer_id, data[0], data[1]);
                fprintf(stderr, "Layer %d Activation: quality=%.4f, error=%.4f, threshold=%.4f->%.4f\n",
                        layer_id, quality, error_for_pid, current, new_threshold);
            }
        }
    } else if (is_predictor_contrib) {
        // ACAS contrib: predictor-knob drift buffer holds [||y_pred_delta||, ||residual||].
        // drift = ||y_pred_delta|| / ||residual|| (the output-contribution of the predictor-skipped
        // neurons relative to the residual). Lower=better, i.e. the SAME control direction as
        // wrong_skip in the audit path, so we feed `drift` exactly where `wrong_skip` goes and drive
        // it toward TARGET_PREDICTOR_CONTRIB (= 1 - ACAS_GATE_TARGET).
        if (data[1] > 0) {  // ||residual|| > 0
            float drift = data[0] / data[1];

            float current = g_sparsity_thresholds.get_predictor_threshold(layer_id);
            float new_threshold;
            if (use_rl) {
                const float push = predictor_rls[layer_id].update(drift);     // drift -> push
                const float threshold_change = push * current * 0.010f;
                new_threshold = std::max(0.1f, current + threshold_change);
            } else {
                BufferedPID & pid = predictor_pids[layer_id];
                float scaled_error = (TARGET_PREDICTOR_CONTRIB - drift) * 10.0f;
                pid.add_sample(scaled_error);
                float adjustment = pid.get_update();
                float threshold_change = adjustment * current * 0.02f;
                new_threshold = std::max(0.1f, current + threshold_change);
            }
            g_sparsity_thresholds.set_predictor_threshold(layer_id, new_threshold);

            // MIMD: predictor runs second on a check token, so adapt the cadence now on the combined
            // error = max(act_err, pred_err) (predictor signal = drift vs TARGET_PREDICTOR_CONTRIB).
            if (g_mimd) g_mimd->adapt(layer_id, std::max(last_act_err_[layer_id], std::fabs(drift - TARGET_PREDICTOR_CONTRIB)));

            const char *log = std::getenv("ACAS_LOG");
            if (log && std::atoi(log) != 0) {
                fprintf(stderr, "Layer %d Predictor: contrib_drift=%.4f (||yd||=%.4f ||res||=%.4f) tgt=%.4f threshold=%.1f->%.1f\n",
                        layer_id, drift, data[0], data[1], TARGET_PREDICTOR_CONTRIB, current, new_threshold);
            }
        }
    } else {
        // Predictor metrics buffer contains [tp, fp, fn, tn]
        float tp = data[0];
        float fp = data[1];
        float fn = data[2];
        float tn = data[3];
        float N = std::max(1.0f, tp + fp + fn + tn);
        float p_true = (tp + fn) / N;
        float fpr = fp / std::max(1.0f, fp + tn);
        float wrong_skip = fpr * (1.0f - p_true);

        // Update predictor threshold: RL (tabular Q-learning) or PID, same gate as activation.
        float current = g_sparsity_thresholds.get_predictor_threshold(layer_id);
        float new_threshold;
        if (use_rl) {
            const float push = predictor_rls[layer_id].update(wrong_skip);   // wrong_skip -> push
            const float threshold_change = push * current * 0.010f;          // SiLU 1% per push-unit
            new_threshold = std::max(0.1f, current + threshold_change);
        } else {
            BufferedPID & pid = predictor_pids[layer_id];
            float scaled_error = (TARGET_QUALITY_PREDICTOR - wrong_skip) * 10.0f;
            pid.add_sample(scaled_error);
            float adjustment = pid.get_update();
            float threshold_change = adjustment * current * 0.02f;
            new_threshold = std::max(0.1f, current + threshold_change);
        }
        g_sparsity_thresholds.set_predictor_threshold(layer_id, new_threshold);

        // MIMD: predictor runs second on a check token, so adapt the cadence now on the combined
        // error = max(act_err, pred_err) (audit predictor signal = wrong_skip vs TARGET_QUALITY_PREDICTOR).
        if (g_mimd) g_mimd->adapt(layer_id, std::max(last_act_err_[layer_id], std::fabs(wrong_skip - TARGET_QUALITY_PREDICTOR)));

        const char *log = std::getenv("ACAS_LOG");
        if (log && std::atoi(log) != 0) {
            fprintf(stderr, "Layer %d Predictor: tp=%.0f fp=%.0f fn=%.0f tn=%.0f p_true=%.4f fpr=%.4f wrong_skip=%.4f tgt=%.4f threshold=%.1f->%.1f\n",
                    layer_id, tp, fp, fn, tn, p_true, fpr, wrong_skip, TARGET_QUALITY_PREDICTOR, current, new_threshold);
        }
    }
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
        while (!metrics_queue.empty()) {
            AsyncQualityMetrics m = metrics_queue.front();
            metrics_queue.pop();
            queue_size = metrics_queue.size();
            lock.unlock();
            apply_update(m);
            processed_count++;
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
