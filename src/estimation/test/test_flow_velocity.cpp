#include "../include/estimator.hpp"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

void read_data_from_disk(double timestamp, cv::Mat &im0, cv::Mat &im1,
                         std::string &flowdata) {
  std::string filename = "/tmp/" + std::to_string(timestamp);
  im0 = cv::imread(filename + "_frame0.png", cv::IMREAD_GRAYSCALE);
  im1 = cv::imread(filename + "_frame1.png", cv::IMREAD_GRAYSCALE);
  std::ifstream flowdatafile(filename + "_flowinfo.txt");
  std::stringstream buffer;
  buffer << flowdatafile.rdbuf();
  flowdata = buffer.str();
}

double get_timestamp(std::string f) {
  std::string delimiter = "_";
  std::string token = f.substr(0, f.find(delimiter));
  try {
    return std::stod(token);
  } catch (std::invalid_argument &e) {
    return -1;
  }
}

template<typename T>
void print(T t) { std::cerr << t << std::endl; }

struct DroneData {
  float time, prev_time, height, prev_height, dt;
  Eigen::Vector3f r, omega, drone_omega;
  Eigen::Matrix3f K, R, prev_R, baseTodom, imgTbase;
};

void parseString(const std::string &dataString, DroneData &data) {
  std::istringstream iss(dataString);
  std::string line;

  while (getline(iss, line)) {
    std::istringstream lineStream(line);

    if (line.find("prev_time:") != std::string::npos) {
      lineStream.ignore(10); // Skip 'prev_time:'
      lineStream >> data.prev_time;
    } else if (line.find("prev_height:") != std::string::npos) {
      lineStream.ignore(12); // Skip 'prev_height:'
      lineStream >> data.prev_height;
    } else if (line.find("height:") != std::string::npos) {
      lineStream.ignore(7); // Skip 'height:'
      lineStream >> data.height;
    } else if (line.find("time:") != std::string::npos) {
      lineStream.ignore(5); // Skip 'time:'
      lineStream >> data.time;
    } else if (line.find("K:") != std::string::npos) {
      for (int i = 0; i < 3; ++i) {
        getline(iss, line);
        std::istringstream matrixStream(line);
        for (int j = 0; j < 3; ++j) {
          matrixStream >> data.K(i, j);
        }
      }
    } else if (line.find("cam_R_enu:") != std::string::npos) {
      int nchars = 10; // Skip 'cam_R_enu:'
      lineStream.ignore(nchars);
      lineStream >> data.R(0, 0);
      lineStream >> data.R(0, 1);
      lineStream >> data.R(0, 2);

      for (int i = 1; i < 3; ++i) {
        getline(iss, line);
        std::istringstream matrixStream(line);
        for (int j = 0; j < 3; ++j) {
          matrixStream >> data.R(i, j);
        }
      }
    } else if (line.find("prev_R:") != std::string::npos) {
      int nchars = 8;
      lineStream.ignore(nchars); // Skip 'prev_R:'
      lineStream >> data.prev_R(0, 0);
      lineStream >> data.prev_R(0, 1);
      lineStream >> data.prev_R(0, 2);

      for (int i = 1; i < 3; ++i) {
        getline(iss, line);
        std::istringstream matrixStream(line);
        for (int j = 0; j < 3; ++j) {
          matrixStream >> data.prev_R(i, j);
        }
      }
    }
  }

  data.dt = data.time - data.prev_time;
}

TEST(TestFlowVelocity, ImageJacobianGivesCorrectResult) {
  cv::Ptr<cv::DISOpticalFlow> disflow = cv::DISOpticalFlow::create(2);
  const double T0 = 470.517327;
  const double T1 = 536.102515;
  std::vector<std::vector<double>> meas_vel;
  std::vector<double> meas_time;
  double ts_spot = -1;
  std::vector<std::string> saved;
  for (const auto &entry :
       std::filesystem::directory_iterator("/tmp")) {
    saved.push_back(entry.path().filename());
  }
  std::sort(saved.begin(), saved.end());

  for (size_t iter = 0; iter < saved.size(); iter++) {
    double tmp = get_timestamp(saved[iter]);
    if (tmp == -1) {
      continue;
    }
    if (ts_spot == -1) {
      ts_spot = tmp;
    }
    if (tmp == ts_spot) {
      continue;
    }
    ts_spot = tmp;
    if (ts_spot < T0 || ts_spot > T1) {
      continue;
    }
    std::cerr << "Parsing ts: " << ts_spot << std::endl;

    cv::Mat im0, im1;
    std::string flowdata;
    read_data_from_disk(ts_spot, im0, im1, flowdata);

    if (im0.empty() || im1.empty()) {
      ASSERT_TRUE(false) << "Could not read images from disk";
    }

    DroneData dronedata{};
    parseString(flowdata, dronedata);
    ASSERT_TRUE(dronedata.time != 0);
    ASSERT_TRUE(dronedata.prev_time != 0);
    ASSERT_TRUE(dronedata.height != 0);
    ASSERT_TRUE(dronedata.prev_height != 0);
    ASSERT_TRUE(dronedata.K != Eigen::Matrix3f::Zero());
    ASSERT_TRUE(dronedata.prev_R != Eigen::Matrix3f::Zero());
    ASSERT_TRUE(dronedata.R != Eigen::Matrix3f::Zero());

    auto est_config = EstimatorConfig{
        .spatial_vel_flow_error = 1,
        .flow_vel_rejection_perc = 0.5};
    auto estimator = Estimator(est_config);

    cv::Mat flow;
    disflow->calc(im0, im1, flow);

    Eigen::VectorXf v_enu = estimator.computeCameraVelocity(
        flow, dronedata.K, dronedata.prev_R,
        dronedata.prev_height, dronedata.dt);
    // print(v_enu.transpose());

    ASSERT_NEAR(v_enu(0), -0.136, 1e-1);
    ASSERT_NEAR(v_enu(1), 0.814, 1e-1);
    ASSERT_NEAR(v_enu(2), -0.177, 1e-1);
    break;

    if (ts_spot > 473) {
      break;
    }
  }
}

TEST(TestFlowVelocityParts, ImageJacobianGivesCorrectResult) {
  const int NUMBER_OF_PIXELS = 3;

  Eigen::MatrixXf J;
  Eigen::MatrixXf uv = Eigen::MatrixXf(2, NUMBER_OF_PIXELS);
  Eigen::VectorXf depth = Eigen::VectorXf(NUMBER_OF_PIXELS);
  Eigen::Matrix3f K = Eigen::Matrix3f::Identity();

  depth << 31.15032, 32.37518, 31.58668;
  K << 285.0, 0.0, 320.0,
      0.0, 285.0, 240.0,
      0.0, 0.0, 1.0;
  uv << 324.47339, 324.02873, 316.40855,
      256.61204, 235.51759, 269.08927;

  Estimator::visjac_p(uv, depth, K, J);

  auto Jtrue = Eigen::MatrixXf(6, 6);
  Jtrue << -9.14918, 0.0, 0.14361, 0.26074, -285.07021, 16.61204,
      0.0, -9.14918, 0.53329, 285.96828, -0.26074, -4.47339,
      -8.80304, 0.0, 0.12444, -0.06336, -285.05695, -4.48241,
      0.0, -8.80304, -0.13845, 285.0705, 0.06336, -4.02873,
      -9.02279, 0.0, -0.1137, -0.36657, -285.04526, 29.08927,
      0.0, -9.02279, 0.92093, 287.96907, 0.36657, 3.59145;

  EXPECT_TRUE(J.isApprox(Jtrue, 1e-6));
}

// TEST(TestFlowVelocityParts, VelocityIsComputedCorrectly) {
//   const int NUMBER_OF_PIXELS = 3;
//   Eigen::MatrixXf J;
//   Eigen::MatrixXf uv = Eigen::MatrixXf(2, NUMBER_OF_PIXELS);
//   Eigen::VectorXf depth = Eigen::VectorXf(NUMBER_OF_PIXELS);
//   Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
//   EstimatorConfig config{
//       .spatial_vel_flow_error = 1,
//       .flow_vel_rejection_perc = 0.5};
//   Estimator estimator(config);

//   depth << 31.15032, 32.37518, 31.58668;
//   K << 285.0, 0.0, 320.0,
//       0.0, 285.0, 240.0,
//       0.0, 0.0, 1.0;
//   uv << 324.47339, 324.02873, 316.40855,
//       256.61204, 235.51759, 269.08927;

//   estimator.visjac_p(uv, depth, K, J);

//   Eigen::VectorXf flow(6);
//   flow << 9.00558, 8.6159, 8.6786, 8.94149, 9.13649, 8.10185;
//   Eigen::VectorXf vel;
//   // estimator.RANSAC_vel_regression(J, flow, Eigen::Matrix3f::Identity(), vel);

//   // Eigen::VectorXf veltrue(6);
//   // veltrue << -1, -1, -1, 0.0, 0.0, 0.0;
//   // EXPECT_TRUE(vel.isApprox(veltrue, 1e-3)) << "vel: " << vel << std::endl
//   //                                          << "veltrue: " << veltrue << std::endl;
// }

// TEST(TestFlowVelocityParts, TestPixelDepthIsCorrectlyComputed) {
//   EstimatorConfig config{
//       .spatial_vel_flow_error = 1,
//       .flow_vel_rejection_perc = 0.5};
//   Estimator estimator(config); // Use the mock estimator

//   Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
//   K << 285.0, 0.0, 320.0,
//       0.0, 285.0, 240.0,
//       0.0, 0.0, 1.0;

//   {
//     Eigen::Matrix3f cam_R_enu = Eigen::Matrix3f::Zero();
//     cam_R_enu << 1.0, 0.0, 0.0,
//         0.0, 0.0, -1.0,
//         0.0, 1.0, 0.0;
//     Eigen::Vector2f pixel = Eigen::Vector2f(323.57283, 255.75045);
//     float height = 20.0;

//     float Z = estimator.get_pixel_z_in_camera_frame(pixel, cam_R_enu, K, height);
//     EXPECT_NEAR(Z, -361.89433, 1e-3);
//   }
//   {
//     // camera Z facing up, aligned with ENUs Z
//     Eigen::Matrix3f cam_R_enu = Eigen::Matrix3f::Identity();
//     // center pixel
//     Eigen::Vector2f pixel(320.0, 240.0);
//     float height = 20.0;

//     float Z = estimator.get_pixel_z_in_camera_frame(pixel, cam_R_enu, K, height);
//     // the point is on the ground, so Z should be negative
//     EXPECT_NEAR(Z, -height, 1e-3);
//   }
//   {
//     // camera Z facing down
//     Eigen::Matrix3f cam_R_enu = Eigen::Matrix3f::Zero();
//     cam_R_enu << 1.0, 0.0, 0.0,
//         0.0, 1.0, 0.0,
//         0.0, 0.0, -1.0;
//     // center pixel
//     Eigen::Vector2f pixel(320.0, 240.0);
//     float height = 20.0;

//     float Z = estimator.get_pixel_z_in_camera_frame(pixel, cam_R_enu, K, height);
//     EXPECT_NEAR(Z, height, 1e-3);
//   }

//   // TODO: should get this depth map out of the experiment and try to plot it!
//   Eigen::Matrix3f cam_R_enu;
//   cam_R_enu << 0.499455, -0.624689, 0.600257,
//       -0.866339, -0.360991, 0.34517,
//       0.00106359, -0.692423, -0.721491;
//   float height = 6.30986;

//   // get_pixel_z_in_camera_frame(pixel, cam_R_enu, K, height)

//   // iterate all the pixels
//   for (int i = 0; i < 640; i++) {
//     for (int j = 0; j < 480; j++) {
//       Eigen::Vector2f pixel(i, j);
//       auto P = estimator.target_position(pixel, cam_R_enu, K, height);
//       EXPECT_NEAR(P[2], -height, 1e-3);
//     }
//   }
// }

// TEST(TestFlowVelocityParts, RansacIsAbleToFilterOutDominantFlow) {
//   const int NUMBER_OF_PIXELS = 4;
//   Eigen::MatrixXf uv = Eigen::MatrixXf(2, NUMBER_OF_PIXELS);
//   Eigen::VectorXf depth = Eigen::VectorXf(NUMBER_OF_PIXELS);
//   Eigen::VectorXf flow(2 * NUMBER_OF_PIXELS);
//   Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
//   EstimatorConfig config{
//       .spatial_vel_flow_error = 2,
//       .flow_vel_rejection_perc = .001};
//   Estimator estimator(config);

//   depth << 31.15032, 32.37518, 31.58668, 31.58668;
//   K << 285.0, 0.0, 320.0,
//       0.0, 285.0, 240.0,
//       0.0, 0.0, 1.0;
//   uv << 324.47339, 324.02873, 316.40855, 200,
//       256.61204, 235.51759, 269.08927, 200;
//   flow << 9.00558, 8.6159, 8.6786, 8.94149, 9.13649, 8.10185,
//       -10, -10; // add some outliers

//   Eigen::MatrixXf J;
//   estimator.visjac_p(uv, depth, K, J);

//   Eigen::VectorXf veltrue(6);
//   veltrue << -1, -1, -1, 0.0, 0.0, 0.0;

//   for (int i = 0; i < 5; i++) {
//     Eigen::VectorXf vel;
//     bool success = estimator.RANSAC_vel_regression(J, flow, vel);
//     EXPECT_TRUE(success);
//     EXPECT_TRUE(vel.isApprox(veltrue, 1e-3)) << "vel: " << vel << std::endl
//                                              << "veltrue: " << veltrue << std::endl;
//   }
// }

// // TODO:
// TEST(TestFlowVelocityParts, PureRotationGivesZeroVelocity) {
// }

// // TEST(TestFlowVelocityParts, FullVelocityOnRealData) {
// //     auto estimator = Estimator();
// //     std::string dir = TEST_DIR;

// //     cv::Mat frame0;
// //     frame0 = cv::imread(dir + "frame0.png", cv::IMREAD_COLOR);
// //     cv::cvtColor(frame0, frame0, cv::COLOR_BGR2RGB);

// //     cv::Mat frame1;
// //     frame1 = cv::imread(dir + "frame1.png", cv::IMREAD_COLOR);
// //     cv::cvtColor(frame1, frame1, cv::COLOR_BGR2RGB);

// // //    cv::imshow("frame", frame0);
// // //    cv::waitKey(0);
// // //    cv::imshow("frame", frame1);
// // //    cv::waitKey(0);

// //     float t0 = 508.652;
// //     float t1 = 508.685;
// //     Eigen::Matrix3f cam_R_enu;
// //     cam_R_enu << 0.499455, -0.624689, 0.600257,
// //             -0.866339, -0.360991, 0.34517,
// //             0.00106359, -0.692423, -0.721491;
// //     float height = 6.30986;
// //     Eigen::Vector3f r;
// //     r << 0.127711, 0.00500937, -0.0735654;
// //     Eigen::Vector3f cam_omega;
// //     cam_omega << 0.09909591711111113, 0.025986013, -0.030931112611111106;
// //     Eigen::Vector3f drone_omega;
// //     drone_omega << -0.03847748836363636, -0.10718755563636365, 0.0076125204545454545;
// //     Eigen::Vector3f gt_vel_ned = {1.282000065, 3.300000191, 0.229000017};

// //     Eigen::Matrix3f K;
// //     K << 385.402, 0.0, 322.133,
// //             0.0, 384.882, 240.013,
// //             0.0, 0.0, 1.0;

// // //     // first call simply record the previous frame
// // //     auto ret = estimator.update_flow_velocity(
// // //             frame0, t0, cam_R_enu, r, K, height, cam_omega, drone_omega);
// // //     EXPECT_TRUE(ret.isApprox(Eigen::Vector3f::Zero(), 1e-6));

// // //     // calculate velocity
// // //     ret = estimator.update_flow_velocity(
// // //             frame1, t1, cam_R_enu, r, K, height, cam_omega, drone_omega);

// // //     Eigen::Matrix3f ned_R_enub = Eigen::Matrix3f::Identity();
// // //     ned_R_enub << 0.0, 1.0, 0.0,
// // //             1.0, 0.0, 0.0,
// // //             0.0, 0.0, -1.0;
// // //     Eigen::Vector3f gt_vel_enu = ned_R_enub * gt_vel_ned;

// // //     auto ret_xy = ret.segment(0, 2);
// // //     auto gt_xy = gt_vel_enu.segment(0, 2);

// // //     EXPECT_NEAR(ret_xy.norm(), gt_xy.norm(), 2e-1);

// // //     EXPECT_TRUE(ret_xy.isApprox(gt_xy, 1e-1))
// // //                         << "computed vel: " << std::endl << ret << std::endl
// // //                         << "ground truth: " << std::endl << gt_vel_enu << std::endl;

// // // Estimator::update_flow_velocity(frame, time, cam_R_enu, &r, &K, height, omega)
// // //  to test this I'll need:
// // //      1) X two frames
// // //      2) X associated time stamps
// // //      3) X cam_R_enu
// // //      4) PJ averaged angular velocity from camera gyro
// // //      5) X height
// // //      6) X arm length r - it's fixed
// // //      7) PJ ground truth velocity
// // //      8) PJ averaged angular velocity of the drone
// // // X - dump from the function
// // // PJ - get from PJ

// // }

// // static int count{0};
// // count++;
// // if (count > 1) {
// //   std::string time_str = std::to_string(time);
// //   cv::imwrite("/tmp/" + time_str + "_frame0.png", *prev_frame_);
// //   cv::imwrite("/tmp/" + time_str + "_frame1.png", frame);
// //   std::ofstream file("/tmp/" + time_str + "_flowinfo.txt");
// //   file << "time:" << time << std::endl;
// //   file << "prev_time:" << pre_frame_time_ << std::endl;
// //   file << "cam_R_enu:" << cam_R_enu << std::endl;
// //   file << "height:" << get_height() << std::endl;
// //   file << "r:" << r << std::endl;
// //   file << "K:" << std::endl
// //        << K << std::endl;
// //   file << "omega:" << omega << std::endl;
// //   file << "drone_omega:" << drone_omega << std::endl;
// //   file << "prev_R:" << prev_cam_R_enu_ << std::endl;
// // }

// for (int i = 0; i < 3; i++) {
//   if (std::abs(omega[i]) > 0.3 || std::abs(drone_omega[i]) > 0.3) {
//     store_flow_state(frame, time, cam_T_enu);
//     return {0, 0, 0};
//   }
// }

//     int every_nth = 16;
// std::vector<cv::Point2f> flow_vecs;
// flow_vecs.reserve(frame.rows * frame.cols / (every_nth * every_nth));
// std::vector<cv::Point> samples;
// samples.reserve(frame.rows * frame.cols / (every_nth * every_nth));
// // the multiplies are orientation dependant
// for (int row = (int) every_nth; row < (frame.rows - every_nth); row += every_nth) {
//   for (int col = (int) every_nth; col < (frame.cols - every_nth); col += every_nth) {
//     // Get the flow from `flow`, which is a 2-channel matrix
//     const cv::Point2f &fxy = flow.at<cv::Point2f>(row, col);
//     flow_vecs.push_back(fxy);
//     samples.emplace_back(row, col);
//   }
// }

// const float MAX_Z = 200;
// Eigen::VectorXf depth(samples.size());
// Eigen::MatrixXf uv = Eigen::MatrixXf(2, samples.size());
// Eigen::VectorXf flow_eigen(2 * flow_vecs.size());
// long insert_idx = 0;
// for (size_t i = 0; i < samples.size(); i++) {
//   const bool is_flow_present = (flow_vecs[i].x != 0 && flow_vecs[i].y != 0);
//   if (!is_flow_present)
//     continue;

//   const float Z = get_pixel_z_in_camera_frame(
//       Eigen::Vector2f(samples[i].x, samples[i].y), prev_cam_T_enu_, K);
//   if (Z < 0 || Z > MAX_Z || std::isnan(Z))
//     continue;

//   depth(insert_idx) = Z;
//   uv(0, insert_idx) = static_cast<float>(samples[i].x);
//   uv(1, insert_idx) = static_cast<float>(samples[i].y);
//   flow_eigen(2 * insert_idx) = flow_vecs[i].x / dt;
//   flow_eigen(2 * insert_idx + 1) = flow_vecs[i].y / dt;
//   ++insert_idx;
// }
// if (insert_idx < 3) {
//   store_flow_state(frame, time, cam_T_enu);
//   return {0, 0, 0};
// }

// // resize the matrices to fit the filled values
// depth.conservativeResize(insert_idx);
// uv.conservativeResize(Eigen::NoChange, insert_idx);
// flow_eigen.conservativeResize(2 * insert_idx);

// Eigen::MatrixXf J; // Jacobian
// visjac_p(uv, depth, K, J);

// // for (long i = 0; i < J.rows(); i++) {
// //   Eigen::Vector3f Jw = {J(i, 3), J(i, 4), J(i, 5)};
// //   flow_eigen(i) -= Jw.dot(omega);
// //   // Eigen::Vector3f Jv = {J(i, 0), J(i, 1), J(i, 2)};
// //   // flow_eigen(i) -= Jv.dot(drone_omega.cross(r));
// // }
// Eigen::VectorXf cam_vel_est;
// // bool success = RANSAC_vel_regression(J.block(0, 0, J.rows(), 3), flow_eigen, cam_vel_est);
// bool success = RANSAC_vel_regression(J, flow_eigen, cam_vel_est);

// // const Eigen::Vector3f v_base = img_T_base.rotation() * cam_vel_est.segment(0, 3)
// //     - drone_omega.cross(img_T_base.translation());
// // const Eigen::Vector3f v_enu = base_T_odom.rotation() * v_base;

// Eigen::Vector3f v_enu = cam_T_enu.rotation() * cam_vel_est.segment(0, 3);