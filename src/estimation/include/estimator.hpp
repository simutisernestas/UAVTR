#pragma once

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "kalman.hpp"

#include <cassert>

template <typename T, size_t S>
class LowPassFilter
{
public:
    LowPassFilter(const std::array<T, S> &b_coefficients, const std::array<T, S> &a_coefficients)
        : b_coefficients_(b_coefficients), a_coefficients_(a_coefficients),
          input_samples_(b_coefficients.size()), filter_buffer_(b_coefficients.size() - 1)
    {
        for (size_t i = 0; i < input_samples_.size(); i++)
            input_samples_[i] = 0.0;
        for (size_t i = 0; i < filter_buffer_.size(); i++)
            filter_buffer_[i] = 0.0;
    }

    double filter(double input)
    {
        assert(input_samples_.size() == b_coefficients_.size());
        assert(filter_buffer_.size() == a_coefficients_.size() - 1);

        double output = 0.0;

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
    void push_pop(std::deque<double> &d, double val)
    {
        d.push_front(val);
        d.pop_back();
    }

    std::array<T, S> b_coefficients_;
    std::array<T, S> a_coefficients_;
    std::deque<double> input_samples_;
    std::deque<double> filter_buffer_;
};

class Estimator
{
public:
    Estimator();

    Eigen::Vector3d compute_pixel_rel_position(
        const Eigen::Vector2d &bbox_c, const Eigen::Matrix3d &cam_R_enu,
        const Eigen::Matrix3d &K, const double height, bool update = true);

    void update_flow_velocity(cv::Mat &frame, const Eigen::Matrix3d &cam_R_enu,
                              const Eigen::Matrix3d &K, const double height);

    void update_imu_accel(const Eigen::Vector3d &accel);

    inline Eigen::VectorXd state() const { return kf_->state(); };

private:
    std::unique_ptr<cv::Mat> prev_frame_{nullptr};
    std::vector<cv::Point2f> p0_, p1_;
    std::unique_ptr<KalmanFilter> kf_;

    typedef LowPassFilter<double, 3> LPF;
    std::array<std::unique_ptr<LPF>, 3> lp_acc_filter_arr_;
};