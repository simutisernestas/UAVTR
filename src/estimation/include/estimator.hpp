#pragma once

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include <opencv2/optflow.hpp>
#include "kalman.hpp"
#include "lpfilter.hpp"

class Estimator {
public:
    Estimator();
    ~Estimator();

    Eigen::Vector3f compute_pixel_rel_position(
            const Eigen::Vector2f &bbox_c, const Eigen::Matrix3f &cam_R_enu,
            const Eigen::Matrix3f &K);

    Eigen::Vector3f update_flow_velocity(cv::Mat &frame, double time, const Eigen::Matrix3f &cam_R_enu,
                                         const Eigen::Vector3f &r, const Eigen::Matrix3f &K,
                                         const Eigen::Vector3f &omega, const Eigen::Vector3f &drone_omega);

    void update_imu_accel(const Eigen::Vector3f &accel, double time);

    void update_height(float height);

    [[nodiscard]] inline Eigen::VectorXf state() const { return kf_->state(); };

    [[nodiscard]] Eigen::MatrixXf covariance() const { return kf_->covariance(); };

    static void visjac_p(const Eigen::MatrixXf &uv,
                         const Eigen::VectorXf &depth,
                         const Eigen::Matrix3f &K,
                         Eigen::MatrixXf &L);

    [[nodiscard]] inline float get_height() const {
        return -kf_->state()[2];
    }

    [[nodiscard]] float get_pixel_z_in_camera_frame(
            const Eigen::Vector2f &pixel, const Eigen::Matrix3f &cam_R_enu,
            const Eigen::Matrix3f &K) const {
        Eigen::Matrix<float, 3, 3> Kinv = K.inverse();
        Eigen::Vector3f lr{0, 0, -1};
        Eigen::Vector3f Puv_hom{pixel[0], pixel[1], 1};
        Eigen::Vector3f Pc = Kinv * Puv_hom;
        Eigen::Vector3f ls = cam_R_enu * (Pc / Pc.norm());
        float d = get_height() / (lr.transpose() * ls);
        Eigen::Vector3f Pt = ls * d;
        // transform back to camera frame
        Pt = cam_R_enu.inverse() * Pt;
        return Pt[2];
    }

    void update_cam_imu_accel(const Eigen::Vector3f &accel, const Eigen::Vector3f &omega,
                              const Eigen::Matrix3f &imu_R_enu, const Eigen::Vector3f &arm);

private:
    static void get_A(Eigen::MatrixXf &A, double dt);

    std::unique_ptr<KalmanFilter> kf_;
    typedef LowPassFilter<float, 3> LPF;
    std::array<std::unique_ptr<LPF>, 3> lp_acc_filter_arr_;
    std::shared_ptr<cv::Mat> prev_frame_{nullptr};
    double pre_frame_time_{-1};
    double pre_imu_time_{-1};
    cv::Ptr<cv::DenseOpticalFlow> optflow_;
    std::atomic<float> latest_height_{0};
};