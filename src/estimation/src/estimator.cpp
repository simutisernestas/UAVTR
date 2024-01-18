#include "estimator.hpp"
#include "eigen_ridge.hpp"
#include <cassert>
#include <chrono>
#include <fstream>
#include <random>
#include <vector>

struct StateUpdateFreq {
  uint64_t count;
  long t0; // milliseconds
  long t1; // milliseconds
};
auto state_update_freq_map = std::map<std::string, StateUpdateFreq>{};

#define KF_STATE_DIM 14
#define KF_MEAS_DIM 9

void record_state_update(const std::string &name) {
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
  if (state_update_freq_map.find(name) == state_update_freq_map.end()) {
    state_update_freq_map[name] = StateUpdateFreq{0, time, -1};
  }
  state_update_freq_map[name].count++;
  state_update_freq_map[name].t1 = time;
}

Estimator::Estimator(EstimatorConfig config) : config_(config) {
  const double dt = 1.0 / 128.0;
  Eigen::MatrixXf A(KF_STATE_DIM, KF_STATE_DIM);
  get_A(A, dt);
  std::cout << "A:" << std::endl
            << A << std::endl;

  assert(config_.Q.size() == KF_STATE_DIM * KF_STATE_DIM); // row major vector
  Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(KF_STATE_DIM, KF_STATE_DIM);
  for (int row = 0; row < KF_STATE_DIM; row++) {
    for (int col = 0; col < KF_STATE_DIM; col++)
      Q(row, col) = config_.Q[row * KF_STATE_DIM + col];
  }
  std::cout << "Q:" << std::endl
            << Q << std::endl;

#define R_POS_DIM 2
  // relative position measurement
  assert(config_.R_pos.size() == R_POS_DIM * R_POS_DIM); // row major vector
  Eigen::MatrixXf C = Eigen::MatrixXf::Zero(R_POS_DIM, KF_STATE_DIM);
  C.block(0, 0, R_POS_DIM, R_POS_DIM) =
      Eigen::MatrixXf::Identity(R_POS_DIM, R_POS_DIM);
  Eigen::MatrixXf R = Eigen::MatrixXf::Zero(R_POS_DIM, R_POS_DIM);
  for (int row = 0; row < R_POS_DIM; row++) {
    for (int col = 0; col < R_POS_DIM; col++)
      R(row, col) = config_.R_pos[row * R_POS_DIM + col];
  }
  std::cout << "R_pos:" << std::endl
            << R << std::endl;
#undef R_POS_DIM

#define R_VEL_DIM 3
  assert(config_.R_vel.size() == R_VEL_DIM * R_VEL_DIM); // row major vector
  R_vel_ = Eigen::MatrixXf::Zero(R_VEL_DIM, R_VEL_DIM);
  for (int row = 0; row < R_VEL_DIM; row++) {
    for (int col = 0; col < R_VEL_DIM; col++)
      R_vel_(row, col) = config_.R_vel[row * R_VEL_DIM + col];
  }
  std::cout << "R_vel:" << std::endl
            << R_vel_ << std::endl;
#undef R_VEL_DIM

#define R_ACC_DIM 3
  assert(config_.R_acc.size() == R_ACC_DIM * R_ACC_DIM); // row major vector
  R_acc_ = Eigen::MatrixXf::Zero(R_ACC_DIM, R_ACC_DIM);
  for (int row = 0; row < R_ACC_DIM; row++) {
    for (int col = 0; col < R_ACC_DIM; col++)
      R_acc_(row, col) = config_.R_acc[row * R_ACC_DIM + col];
  }
  std::cout << "R_acc:" << std::endl
            << R_acc_ << std::endl;
#undef R_ACC_DIM

  Eigen::MatrixXf P =
      Eigen::MatrixXf::Identity(KF_STATE_DIM, KF_STATE_DIM) * 10000.0;
  kf_ = std::make_unique<KalmanFilter>(A, C, Q, R, P);

  optflow_ = cv::DISOpticalFlow::create(2);
}

Estimator::~Estimator() {
  // log out the frequency of state updates to a file
  std::ofstream file("/tmp/state_update_freq.txt");

  for (const auto &pair : state_update_freq_map) {
    const auto &name = pair.first;
    const auto &freq = pair.second;
    auto dt = static_cast<double>(freq.t1 - freq.t0);
    auto freq_hz = static_cast<double>(freq.count) / dt * 1000.0;
    file << name << " " << freq_hz << " Hz" << std::endl;
  }

  file.close();
}

void Estimator::get_A(Eigen::MatrixXf &A, double dt) {
  auto ddt2 = static_cast<float>(dt * dt * .5);
  assert(dt > 0 && dt < 1);

  A = Eigen::MatrixXf::Identity(KF_STATE_DIM, KF_STATE_DIM);
  A.block(0, 3, 3, 3) = Eigen::Matrix3f::Identity() * -dt;
  A.block(0, 6, 2, 2) = Eigen::Matrix2f::Identity() * dt;
  A.block(0, 8, 3, 3) = Eigen::Matrix3f::Identity() * -ddt2;
  A.block(3, 8, 3, 3) = Eigen::Matrix3f::Identity() * dt;
}

Eigen::Vector3f Estimator::update_target_position(
    const Eigen::Vector2f &bbox_c, const EigenAffine &cam_T_enu,
    const Eigen::Matrix3f &K, const Eigen::Vector3f &t) {
  float height = get_height();
  if (height < 1.0f)
    return {0, 0, 0};

  Eigen::Vector3f Pt = target_position(bbox_c, cam_T_enu, K, height);

  if (kf_->is_initialized()) {
    Eigen::Vector2f xy_meas(2);
    xy_meas << Pt[0], Pt[1];
    kf_->update(xy_meas);
    record_state_update(__FUNCTION__);
  } else {
    Eigen::VectorXf x0 = Eigen::VectorXf::Zero(KF_STATE_DIM);
    for (int i = 0; i < 3; i++)
      x0[i] = Pt[i];
    kf_->init(x0);
  }

  target_last_seen_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
  return Pt;
}

void Estimator::update_height(const float height) {
  latest_height_.store(height);

  if (!kf_->is_initialized())
    return;

  Eigen::MatrixXf C_height = Eigen::MatrixXf::Zero(1, KF_STATE_DIM);
  C_height(0, 2) = 1.0;
  Eigen::MatrixXf R = Eigen::MatrixXf::Identity(1, 1);
  R *= 2.0;

  // TODO: this might need flipping
  Eigen::VectorXf h(1);
  h << -height;
  // the relative height is negative
  kf_->update(h, C_height, R);

  record_state_update(__FUNCTION__);
}

void Estimator::update_imu_accel(const Eigen::Vector3f &accel, double time) {
  if (!kf_->is_initialized())
    return;
  if (pre_imu_time_ < 0) {
    pre_imu_time_ = time;
    return;
  }
  auto dt = time - pre_imu_time_;
  assert(dt > 0);
  assert(dt < 1.0);
  pre_imu_time_ = time;

  Eigen::MatrixXf C_accel = Eigen::MatrixXf::Zero(3, KF_STATE_DIM);
  C_accel.block(0, 8, 3, 3) = Eigen::Matrix3f::Identity();
  C_accel.block(0, 11, 3, 3) = Eigen::Matrix3f::Identity();
  kf_->update(accel, C_accel, R_acc_);

  Eigen::MatrixXf A(KF_STATE_DIM, KF_STATE_DIM);
  get_A(A, dt);
  kf_->predict(A);

  int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
  if (now - target_last_seen_ > 1000) {
    kf_->reset_boat_velocity();
  }

  record_state_update(__FUNCTION__);
}

void Estimator::update_cam_imu_accel(const Eigen::Vector3f &accel, const Eigen::Vector3f &omega,
                                     const Eigen::Matrix3f &imu_R_enu, const Eigen::Vector3f &arm) {
  return; // TODO:
  if (!kf_->is_initialized())
    return;

  Eigen::Vector3f accel_enu = imu_R_enu * accel;
  // subtract gravity
  accel_enu[2] -= 9.81;
  Eigen::Vector3f omega_enu = imu_R_enu * omega;

  Eigen::Vector3f accel_body = accel_enu - omega_enu.cross(omega_enu.cross(arm));

  Eigen::MatrixXf C_accel(3, KF_STATE_DIM);
  C_accel << 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0;
  Eigen::MatrixXf R_accel(3, 3);
  R_accel << Eigen::Matrix3f::Identity() * 4;
  kf_->update(accel_body, C_accel, R_accel);
}

void Estimator::visjac_p(const Eigen::MatrixXf &uv,
                         const Eigen::VectorXf &depth,
                         const Eigen::Matrix3f &K,
                         Eigen::MatrixXf &L) {
  assert(uv.cols() == depth.size());

  L.resize(depth.size() * 2, 6);
  L.setZero();
  const Eigen::Matrix3f Kinv = K.inverse();

  for (int i = 0; i < uv.cols(); i++) {
    const float z = depth(i);
    const Eigen::Vector3f p(uv(0, i), uv(1, i), 1.0);

    // convert to normalized image-plane coordinates
    const Eigen::Vector3f xy = Kinv * p;
    float x = xy(0);
    float y = xy(1);

    // 2x6 Jacobian for this point
    Eigen::Matrix<float, 2, 6> Lp;
    Lp << -1 / z, 0.0, x / z, x * y, -(1 + x * x), y,
        0.0, -1 / z, y / z, 1 + y * y, -x * y, -x;
    Lp = K.block(0, 0, 2, 2) * Lp;

    // push into Jacobian
    L.block(2 * i, 0, 2, 6) = Lp;
  }
}

bool Estimator::RANSACRegression(const Eigen::MatrixXf &J,
                                 const Eigen::VectorXf &flow_vectors,
                                 Eigen::VectorXf &cam_vel_est) {
  // https://rpg.ifi.uzh.ch/docs/Visual_Odometry_Tutorial.pdf slide 68
  // >> outlier_percentage = .75
  // >>> np.log(1 - 0.999) / np.log(1 - (1 - outlier_percentage) ** n_samples)
  // 438.63339476983924
  const size_t n_iterations = 438.63339476983924;
  const size_t n_samples{3}; // minimum required to fit model
  const size_t n_points = flow_vectors.rows() / 2;

  std::random_device rd;                                  // obtain a random number from hardware
  std::minstd_rand gen(rd());                             // seed the generator
  std::uniform_int_distribution<> distr(0, n_points - 1); // define the range

  auto best_inliers = std::vector<size_t>{}; // best inlier indices
  Eigen::MatrixXf J_samples(n_samples * 2, J.cols());
  J_samples.setZero();
  Eigen::VectorXf flow_samples(n_samples * 2);
  flow_samples.setZero();
  Eigen::VectorXf x_est(J.cols());
  std::vector<size_t> inlier_idxs;
  inlier_idxs.reserve(n_points);
  for (size_t iter{0}; iter <= n_iterations; ++iter) {
    // randomly select n_samples from data
    for (size_t i{0}; i < n_samples; ++i) {
      size_t idx = distr(gen);
      // take sampled data
      J_samples.block(i * 2, 0, 2, J.cols()) = J.block(idx * 2, 0, 2, J.cols());
      flow_samples.segment(i * 2, 2) = flow_vectors.segment(idx * 2, 2);
    }
    // solve for velocity
    x_est = J_samples.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(flow_samples);

    Eigen::VectorXf error = J * x_est - flow_vectors;

    // compute inliers
    for (long i{0}; i < static_cast<long>(n_points); ++i) {
      const float error_x = error(i * 2);
      const float error_y = error(i * 2 + 1);
      const float error_norm = std::abs(error_x) + std::abs(error_y);
      if (std::isnan(error_norm) || std::isinf(error_norm)) {
        cam_vel_est = Eigen::VectorXf::Zero(J.cols());
        return false;
      }
      if (error_norm < config_.spatial_vel_flow_error) { // in pixels
        inlier_idxs.push_back(i);
      }
    }

    if (best_inliers.size() < inlier_idxs.size()) {
      best_inliers = inlier_idxs;
    }

    if (static_cast<float>(best_inliers.size()) > 0.5f * static_cast<float>(n_points))
      break;

    inlier_idxs.clear();
  }

  std::cout << best_inliers.size() << " inliers out of " << n_points << std::endl;
  if (best_inliers.size() < static_cast<size_t>(static_cast<double>(n_points) * config_.flow_vel_rejection_perc)) {
    cam_vel_est = Eigen::VectorXf::Zero(J.cols());
    return false;
  }

  J_samples.resize(best_inliers.size() * 2, J.cols());
  flow_samples.resize(best_inliers.size() * 2);
  // solve for best inliers
  for (size_t i{0}; i < best_inliers.size(); ++i) {
    J_samples.block(i * 2, 0, 2, J.cols()) = J.block(best_inliers[i] * 2, 0, 2, J.cols());
    flow_samples.segment(i * 2, 2) = flow_vectors.segment(best_inliers[i] * 2, 2);
  }
  cam_vel_est = J_samples.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(flow_samples);
  return true;
}

void Estimator::store_flow_state(cv::Mat &frame, double time,
                                 const EigenAffine &cam_T_enu) {
  this->pre_frame_time_ = time;
  this->prev_frame_ = std::make_shared<cv::Mat>(frame);
  this->prev_cam_T_enu_ = cam_T_enu;
}

Eigen::Vector3f Estimator::update_flow_velocity(cv::Mat &frame, double time,
                                                const EigenAffine &base_T_odom,
                                                const EigenAffine &img_T_base,
                                                const Eigen::Matrix3f &K, const Eigen::Vector3f &omega,
                                                const Eigen::Vector3f &drone_omega) {
  cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
  EigenAffine cam_T_enu = base_T_odom * img_T_base;
  const double dt = time - pre_frame_time_;
  assert(dt > 0);
  if (!prev_frame_ || dt > .2 || !kf_->is_initialized()) {
    store_flow_state(frame, time, cam_T_enu);
    return {0, 0, 0};
  }

#ifdef TESTWRITE
  static int count{0};
  static float height{0};
  count++;
  if (count > 1) {
    std::string time_str = std::to_string(time);
    cv::imwrite("/tmp/" + time_str + "_frame0.png", *prev_frame_);
    cv::imwrite("/tmp/" + time_str + "_frame1.png", frame);
    std::ofstream file("/tmp/" + time_str + "_flowinfo.txt");
    file << "time:" << time << std::endl;
    file << "prev_time:" << pre_frame_time_ << std::endl;
    file << "cam_R_enu:" << cam_T_enu.rotation() << std::endl;
    file << "height:" << get_height() << std::endl;
    file << "prev_height:" << height << std::endl;
    file << "r:" << img_T_base.translation() << std::endl;
    file << "K:" << std::endl
         << K << std::endl;
    file << "omega:" << omega << std::endl;
    file << "drone_omega:" << drone_omega << std::endl;
    file << "prev_R:" << prev_cam_T_enu_.rotation() << std::endl;
    file << "baseTodom:" << base_T_odom.matrix() << std::endl;
    file << "imgTbase:" << img_T_base.matrix() << std::endl;
  }
  height = get_height();
#endif

  optflow_->calc(*prev_frame_, frame, prev_flow_);

  Eigen::Vector3f v_enu = computeCameraVelocity(
      prev_flow_, K, prev_cam_T_enu_.rotation(), get_height(), dt);

  bool failed = v_enu.isZero(1e-6);
  if (kf_->is_initialized() && !failed) {
    Eigen::MatrixXf C_vel = Eigen::MatrixXf::Zero(3, KF_STATE_DIM);
    C_vel.block(0, 3, 3, 3) = Eigen::Matrix3f::Identity();
    kf_->update(v_enu, C_vel, R_vel_);

    record_state_update(__FUNCTION__);
  }

  store_flow_state(frame, time, cam_T_enu);
  return -v_enu;
}

float Estimator::get_pixel_z_in_camera_frame(
    const Eigen::Vector2f &pixel, const EigenAffine &cam_T_enu,
    const Eigen::Matrix3f &K, float height) const {
  if (height < 0)
    height = get_height();
  Eigen::Vector3f Pt = target_position(pixel, cam_T_enu, K, height);
  // transform back to camera frame
  Pt = cam_T_enu.inverse() * Pt;
  return Pt[2];
}

// function to compute pixel 3D position in ENU frame
// by assumption that the target is on the ground plane
Eigen::Vector3f Estimator::target_position(const Eigen::Vector2f &pixel,
                                           const EigenAffine &cam_T_enu,
                                           const Eigen::Matrix3f &K, float height) const {
  Eigen::Matrix<float, 3, 3> Kinv = K.inverse();
  // vector pointing downwards
  Eigen::Vector3f lr{0, 0, -1};
  // pixel in homogenous coordinates
  Eigen::Vector3f Puv_hom{pixel[0], pixel[1], 1};
  // back-project pixel to camera frame; homogenous coordinate
  Eigen::Vector3f Pc = Kinv * Puv_hom;
  // rotate to ENU frame and normalize
  Eigen::Vector3f ls = cam_T_enu * (Pc / Pc.norm());
  // compute depth from height information
  float d = height / (lr.transpose() * ls);
  // scale unit vector by pixel depth
  Eigen::Vector3f Pt = ls * d;
  return Pt;
}

Eigen::VectorXf Estimator::computeCameraVelocity(
    const cv::Mat &flow, const Eigen::Matrix3f &K,
    const Eigen::Matrix3f &R, float height, float dt) {
  int NTH = 21;
  long size = flow.rows * flow.cols / (NTH * NTH);
  Eigen::MatrixXf pixels(2, size);
  Eigen::VectorXf flows(2 * size);
  Eigen::VectorXf depths(size);
  long insert_idx = 0;
  for (int i = 0; i < flow.rows; i += NTH) {
    for (int j = 0; j < flow.cols; j += NTH) {
      if (insert_idx >= size)
        break;

      cv::Vec2f f = flow.at<cv::Vec2f>(i, j);
      pixels(0, insert_idx) = static_cast<float>(j);
      pixels(1, insert_idx) = static_cast<float>(i);
      flows(2 * insert_idx) = f[0] / dt;
      flows(2 * insert_idx + 1) = f[1] / dt;
      Eigen::Affine3f cam_T_enu(R);
      depths(insert_idx) = get_pixel_z_in_camera_frame(
          pixels.col(insert_idx), cam_T_enu, K, height);
      insert_idx++;
    }
  }
  assert(insert_idx == size);

  Eigen::MatrixXf Jac;
  visjac_p(pixels, depths, K, Jac);

  Eigen::VectorXf vel_omega = Eigen::VectorXf::Zero(6);
  bool success = RANSACRegression(Jac, flows, vel_omega);
  if (!success)
    return Eigen::VectorXf::Zero(3);

  Eigen::VectorXf v_enu = R * vel_omega.segment(0, 3);
  return v_enu;
}

void draw_flow() {
#ifdef DRAW
  //######################    DRAWING
  cv::Mat drawing_frame = frame.clone();
  for (int y = 0; y < drawing_frame.rows; y += every_nth) {
    for (int x = 0; x < drawing_frame.cols; x += every_nth) {
      // Get the flow from `flow`, which is a 2-channel matrix
      const cv::Point2f &fxy = flow.at<cv::Point2f>(y, x);
      // Draw lines on `drawing_frame` to represent flow
      cv::line(drawing_frame, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
               cv::Scalar(0, 255, 0));
      cv::circle(drawing_frame, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
    }
  }
  auto dominant_flow_vec = std::accumulate(flow_vecs.begin(), flow_vecs.end(), cv::Point2f(0, 0));
  dominant_flow_vec.y /= (float) flow_vecs.size();
  dominant_flow_vec.x /= (float) flow_vecs.size();
  // show the dominant flow vector on the image
  cv::line(drawing_frame, cv::Point(drawing_frame.cols / 2, drawing_frame.rows / 2),
           cv::Point(cvRound(drawing_frame.cols / 2 + dominant_flow_vec.x),
                     cvRound(drawing_frame.rows / 2 + dominant_flow_vec.y)),
           cv::Scalar(0, 0, 255), 5);
  // Display the image with vectors
  cv::imshow("Optical Flow Vectors", drawing_frame);
  cv::waitKey(1);
  //######################    DRAWING
#endif
}