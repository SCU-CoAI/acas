#include "pid-controller.h"
#include <cmath>

BufferedPID::Sample::Sample(float e, int s) : error(e), step(s) {}

BufferedPID::BufferedPID() : Kp(0), Ki(0), Kd(0), buffer_size(0), max_age(0), current_step(0) {}

BufferedPID::BufferedPID(float Kp, float Ki, float Kd, size_t buffer_size, int max_age) :
    Kp(Kp),
    Ki(Ki),
    Kd(Kd),
    buffer_size(buffer_size),
    max_age(max_age),
    current_step(0) {}

void BufferedPID::add_sample(float error) {
    current_step++;  // Auto-increment internal step counter
    if (buffer.size() >= buffer_size) {
        buffer.pop_front();
    }
    buffer.emplace_back(error, current_step);
}

float BufferedPID::get_update() {
    std::vector<Sample> valid_samples;
    for (const auto & sample : buffer) {
        if ((current_step - sample.step) <= max_age) {
            valid_samples.push_back(sample);
        }
    }

    if (valid_samples.empty()) {
        return 0.0f;
    }

    float current_error = valid_samples.back().error;

    float integral      = 0.0f;
    for (const auto & sample : valid_samples) {
        integral += sample.error;
    }

    float derivative = 0.0f;
    if (valid_samples.size() >= 2) {
        derivative = current_error - valid_samples[valid_samples.size() - 2].error;
    }

    return (Kp * current_error) + (Ki * integral) + (Kd * derivative);
}

float logit(float x) {
    x = std::max(1e-10f, std::min(1.0f - 1e-10f, x));
    return std::log(x / (1.0f - x));
}

float calculate_error(float p, float target_p, float sensitivity) {
    // ensure p and target_p are within bounds
    p = std::max(1e-10f, std::min(1.0f - 1e-10f, p));
    target_p = std::max(1e-10f, std::min(1.0f - 1e-10f, target_p));

    // calculate the logit difference
    float diff = logit(target_p) - logit(p);

    // sigmoid function to scale the error
    return std::copysign(1.0f, diff) * (2.0f / (1.0f + std::exp(-sensitivity * std::abs(diff))) - 1.0f);
}
