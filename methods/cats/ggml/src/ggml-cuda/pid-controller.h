#ifndef PIDCONTROLLER_H
#define PIDCONTROLLER_H

#include <deque>
#include <vector>
#include <algorithm>
#include <cstddef>  // For size_t

class BufferedPID {
  public:
    struct Sample {
        float error;
        int   step;
        Sample(float e, int s);
    };

    BufferedPID();
    BufferedPID(float Kp, float Ki, float Kd, size_t buffer_size = 5, int max_age = 10);
    void  add_sample(float error);
    float get_update();

  private:
    float              Kp, Ki, Kd;
    size_t             buffer_size;
    int                max_age;
    int                current_step;  // Internal step counter
    std::deque<Sample> buffer;
};

// PID controller error calculation helpers
float logit(float x);
float calculate_error(float p, float target_p, float sensitivity=5.0f);

#endif
