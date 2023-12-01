#pragma once

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include <opencv2/optflow.hpp>
#include "kalman.hpp"

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

class Estimator {
public:
    Estimator();

    Eigen::Vector3f compute_pixel_rel_position(
            const Eigen::Vector2f &bbox_c, const Eigen::Matrix3f &cam_R_enu,
            const Eigen::Matrix3f &K, float height, bool update = true);

    Eigen::Vector3f update_flow_velocity(cv::Mat &frame, double time, const Eigen::Matrix3f &cam_R_enu,
                                         const Eigen::Vector3f &r, const Eigen::Matrix3f &K, float height,
                                         const Eigen::Vector3f &omega, const Eigen::Vector3f &drone_omega);

    void update_imu_accel(const Eigen::Vector3f &accel, double dt);

    void update_height(float height);

    [[nodiscard]] inline Eigen::VectorXf state() const { return kf_->state(); };

    static void visjac_p(const Eigen::MatrixXf &uv,
                         const Eigen::VectorXf &depth,
                         const Eigen::Matrix3f &K,
                         Eigen::MatrixXf &L);

    static void compute_velocity(const Eigen::MatrixXf &J,
                                 const Eigen::VectorXf &flow,
                                 Eigen::VectorXf &vel);

    static float get_pixel_z_in_camera_frame(
            const Eigen::Vector2f &pixel, const Eigen::Matrix3f &cam_R_enu,
            const Eigen::Matrix3f &K, const float height) {
        Eigen::Matrix<float, 3, 3> Kinv = K.inverse();
        Eigen::Vector3f lr{0, 0, -1};
        Eigen::Vector3f Puv_hom{pixel[0], pixel[1], 1};
        Eigen::Vector3f Pc = Kinv * Puv_hom;
        Eigen::Vector3f ls = cam_R_enu * (Pc / Pc.norm());
        float d = height / (lr.transpose() * ls);
        Eigen::Vector3f Pt = ls * d;
        // transform back to camera frame
        Pt = cam_R_enu.inverse() * Pt;
        return Pt[2];
    }

private:
    static void get_A(Eigen::MatrixXf &A, double dt);

    std::unique_ptr<KalmanFilter> kf_;

    typedef LowPassFilter<float, 3> LPF;
    std::array<std::unique_ptr<LPF>, 3> lp_acc_filter_arr_;

    std::shared_ptr<cv::Mat> prev_frame_{nullptr};
    double pre_frame_time_{-1};
    cv::Ptr<cv::DenseOpticalFlow> optflow_;
};