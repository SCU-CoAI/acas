// rlcontroller.h — tabular Q-learning controller for the ACAS threshold/alpha knobs
// (accuracy/code/modeling_*.py). A drop-in alternative to the PID at the controller
// level: update(quality) -> push in decision-boundary units; the CALLER scales the push
//   ReLU (SI/Grasp):  alpha += round(push * alpha_unit),  alpha_unit = 10000/(hidden/2)
//   SiLU (CATS/...):  threshold += push * threshold * 0.010
// The action set is supplied by the call site (default = ReLU set):
//   ReLU (SI/Grasp):  {-8, -3, -1, 0, 1, 3, 8}          (relu_actions(), the default)
//   SiLU (CATS/...):  {-5, -0.7, -0.1, 0, 0.1, 0.7, 5}  (silu_actions(); finer steps for the
//                                                        multiplicative threshold knob)
// Same information as the PID (metric vs target), reward = -|metric - target|. Selected at
// controller init by ACAS_CONTROLLER=rl (else pid). Per-layer instance, seed = layer index.
#pragma once
#include <array>
#include <unordered_map>
#include <deque>
#include <random>
#include <cmath>
#include <algorithm>

class RLController {
public:
    static const int N_ACTIONS = 7;

    // Action sets in decision-boundary units; the call site picks its method's set.
    static const float * relu_actions() {
        static const float A[N_ACTIONS] = {-8.0f, -3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 8.0f};
        return A;
    }
    static const float * silu_actions() {
        static const float A[N_ACTIONS] = {-5.0f, -0.7f, -0.1f, 0.0f, 0.1f, 0.7f, 5.0f};
        return A;
    }

    RLController(float target, int seed = 0,
                float alpha = 0.3f, float gamma = 0.85f,
                float eps0 = 0.30f, float eps_min = 0.02f, int eps_decay_steps = 200,
                const float * actions = nullptr)
        : target_(target), alpha_(alpha), gamma_(gamma),
          eps0_(eps0), eps_min_(eps_min), eps_decay_steps_(eps_decay_steps),
          t_(0), prev_s_(-1), prev_a_(-1), rng_((unsigned int) seed), last_push_(0.0f) {
        const float * src = actions ? actions : relu_actions();
        for (int i = 0; i < N_ACTIONS; ++i) actions_[i] = src[i];
    }

    // convenience: (target, seed, actions) without restating the RL hyperparameter defaults
    RLController(float target, int seed, const float * actions)
        : RLController(target, seed, 0.3f, 0.85f, 0.30f, 0.02f, 200, actions) {}

    // quality = the chosen metric (precision / contrib / l2); returns the push (decision-boundary units)
    float update(float quality) {
        qhist_.push_back(quality);
        if ((int) qhist_.size() > SIGMA_WINDOW) qhist_.pop_front();

        const float e = quality - target_;
        const int s = state_(e);
        std::array<float, N_ACTIONS> & qrow = Q_[s];   // operator[] value-inits to zeros

        if (prev_s_ >= 0) {
            const float r = -std::fabs(e);
            std::array<float, N_ACTIONS> & prow = Q_[prev_s_];
            float maxq = qrow[0];
            for (int i = 1; i < N_ACTIONS; ++i) maxq = std::max(maxq, qrow[i]);
            prow[prev_a_] += alpha_ * (r + gamma_ * maxq - prow[prev_a_]);
        }

        int a;
        std::uniform_real_distribution<float> ur(0.0f, 1.0f);
        if (ur(rng_) < eps_()) {
            std::uniform_int_distribution<int> ui(0, N_ACTIONS - 1);
            a = ui(rng_);
        } else {
            float mx = qrow[0];
            for (int i = 1; i < N_ACTIONS; ++i) mx = std::max(mx, qrow[i]);
            int cand[N_ACTIONS], nc = 0;
            for (int i = 0; i < N_ACTIONS; ++i) if (qrow[i] == mx) cand[nc++] = i;
            std::uniform_int_distribution<int> uc(0, nc - 1);
            a = cand[uc(rng_)];
        }

        // decision-boundary units; the caller scales these into alpha / threshold steps
        prev_s_ = s; prev_a_ = a; ++t_;
        last_push_ = actions_[a];
        return last_push_;
    }

    float last_push() const { return last_push_; }

private:
    static const int   SIGMA_WINDOW   = 40;
    static constexpr float SIGMA_FALLBACK = 0.01f;

    float sigma_() const {
        const int n = (int) qhist_.size();
        if (n < 8) return SIGMA_FALLBACK;
        float m = 0.0f; for (float q : qhist_) m += q; m /= n;
        float var = 0.0f; for (float q : qhist_) var += (q - m) * (q - m); var /= n;
        return std::max(std::sqrt(var), 1e-4f);
    }
    int state_(float e) const {
        const float sig = sigma_();
        int s = 0;
        const int ks[6] = {-3, -2, -1, 1, 2, 3};
        for (int k : ks) if (e > k * sig) ++s;
        return s;     // 0..6
    }
    float eps_() const {
        const float frac = std::min(1.0f, (float) t_ / (float) std::max(1, eps_decay_steps_));
        return eps0_ + (eps_min_ - eps0_) * frac;
    }

    float target_, alpha_, gamma_, eps0_, eps_min_;
    int eps_decay_steps_, t_, prev_s_, prev_a_;
    float actions_[N_ACTIONS];   // call-site-supplied action set (see relu_actions/silu_actions)
    std::unordered_map<int, std::array<float, N_ACTIONS>> Q_;
    std::deque<float> qhist_;
    std::mt19937 rng_;
    float last_push_;
};
