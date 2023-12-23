#pragma once

#include "Eigen/Dense"
#include "kalman.hpp"
#include "lpfilter.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/optflow.hpp>

struct EstimatorConfig {
  float spatial_vel_flow_error;  // px L1 norm
  float flow_vel_rejection_perc; // %%%
};

class Estimator {
public:
  Estimator(EstimatorConfig config);
  ~Estimator();

  Eigen::Vector3f update_target_position(
      const Eigen::Vector2f &bbox_c, const Eigen::Matrix3f &cam_R_enu,
      const Eigen::Matrix3f &K, const Eigen::Vector3f &t);

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
    float h = -kf_->state()[2];
    if (std::abs(h) < 1.0f)
      h = latest_height_.load();
    return h;
  }

  Eigen::Vector3f target_position(const Eigen::Vector2f &pixel, const Eigen::Matrix3f &cam_R_enu, const Eigen::Matrix3f &K, float height) const;

  float get_pixel_z_in_camera_frame(
      const Eigen::Vector2f &pixel, const Eigen::Matrix3f &cam_R_enu,
      const Eigen::Matrix3f &K, float height = -1) const;

  void update_cam_imu_accel(const Eigen::Vector3f &accel, const Eigen::Vector3f &omega,
                            const Eigen::Matrix3f &imu_R_enu, const Eigen::Vector3f &arm);

  bool RANSAC_vel_regression(const Eigen::MatrixXf &J,
                             const Eigen::VectorXf &flow_vectors,
                             Eigen::VectorXf &cam_vel_est);

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
  EstimatorConfig config_;
};