#pragma once

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include <opencv2/optflow.hpp>
#include "kalman.hpp"

#include <cassert>

template<typename T, size_t S>
class LowPassFilter {
public:
    LowPassFilter(const std::array<T, S> &b_coefficients, const std::array<T, S> &a_coefficients)
            : b_coefficients_(b_coefficients), a_coefficients_(a_coefficients),
              input_samples_(b_coefficients.size()), filter_buffer_(b_coefficients.size() - 1) {
        for (double &input_sample: input_samples_)
            input_sample = 0.0;
        for (double &i: filter_buffer_)
            i = 0.0;
    }

    double filter(double input) {
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
    void push_pop(std::deque<double> &d, double val) {
        d.push_front(val);
        d.pop_back();
    }

    std::array<T, S> b_coefficients_;
    std::array<T, S> a_coefficients_;
    std::deque<double> input_samples_;
    std::deque<double> filter_buffer_;
};

class Estimator {
public:
    Estimator();

    Eigen::Vector3d compute_pixel_rel_position(
            const Eigen::Vector2d &bbox_c, const Eigen::Matrix3d &cam_R_enu,
            const Eigen::Matrix3d &K, double height, bool update = true);

    Eigen::Vector3d update_flow_velocity(cv::Mat &frame, double time, const Eigen::Matrix3d &cam_R_enu,
                                         const Eigen::Vector3d &r, const Eigen::Matrix3d &K, double height,
                                         const Eigen::Vector3d &omega, const Eigen::Vector3d &drone_omega);

    void update_imu_accel(const Eigen::Vector3d &accel, double dt);

    void update_height(double height);

    [[nodiscard]] inline Eigen::VectorXd state() const { return kf_->state(); };

    static void visjac_p(const Eigen::MatrixXd &uv,
                         const Eigen::VectorXd &depth,
                         const Eigen::Matrix3d &K,
                         Eigen::MatrixXd &L);

    static void compute_velocity(const Eigen::MatrixXd &J,
                                 const Eigen::VectorXd &flow,
                                 Eigen::VectorXd &vel);

    static double get_pixel_z_in_camera_frame(
            const Eigen::Vector2d &pixel, const Eigen::Matrix3d &cam_R_enu,
            const Eigen::Matrix3d &K, const double height) {
        Eigen::Matrix<double, 3, 3> Kinv = K.inverse();
        Eigen::Vector3d lr{0, 0, -1};
        Eigen::Vector3d Puv_hom{pixel[0], pixel[1], 1};
        Eigen::Vector3d Pc = Kinv * Puv_hom;
        Eigen::Vector3d ls = cam_R_enu * (Pc / Pc.norm());
        double d = height / (lr.transpose() * ls);
        Eigen::Vector3d Pt = ls * d;
        // transform back to camera frame
        Pt = cam_R_enu.inverse() * Pt;
        return Pt[2];
    }

private:
    static void get_A(Eigen::MatrixXd &A, double dt);

    std::unique_ptr<KalmanFilter> kf_;

    typedef LowPassFilter<double, 3> LPF;
    std::array<std::unique_ptr<LPF>, 3> lp_acc_filter_arr_;

    std::shared_ptr<cv::Mat> prev_frame_{nullptr};
    double pre_frame_time_{-1};
    Eigen::Matrix3d prev_cam_R_enu_{};
    cv::Ptr<cv::optflow::DenseRLOFOpticalFlow> optflow_ = cv::optflow::DenseRLOFOpticalFlow::create();
};