#pragma once

#include <array>
#include <deque>
#include <cstddef>

template<typename T, size_t S>
class LowPassFilter {
public:
    LowPassFilter(const std::array<T, S> &b_coefficients, const std::array<T, S> &a_coefficients)
            : b_coefficients_(b_coefficients), a_coefficients_(a_coefficients),
              input_samples_(b_coefficients.size()), filter_buffer_(b_coefficients.size() - 1) {
        for (float &input_sample: input_samples_)
            input_sample = 0.0;
        for (float &i: filter_buffer_)
            i = 0.0;
    }

    float filter(float input) {
        float output = 0.0;

        push_pop(input_samples_, input);

        // compute the new output
        for (size_t i = 0; i < filter_buffer_.size(); i++)
            output -= a_coefficients_[i + 1] * filter_buffer_[i];
        for (size_t i = 0; i < input_samples_.size(); i++)
            output += b_coefficients_[i] * input_samples_[i];

        push_pop(filter_buffer_, output);

        return output;
    }

private:
    void push_pop(std::deque<float> &d, float val) {
        d.push_front(val);
        d.pop_back();
    }

    std::array<T, S> b_coefficients_;
    std::array<T, S> a_coefficients_;
    std::deque<float> input_samples_;
    std::deque<float> filter_buffer_;
};