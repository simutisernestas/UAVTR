// #include <eigen3/Eigen/Core>
// #include <opencv4/opencv2/opencv.hpp>
// #include "gtest/gtest.h"
// #include "../include/estimator.hpp"

// TEST(TestFlowVelocityParts, ImageJacobianGivesCorrectResult) {
//     const int NUMBER_OF_PIXELS = 3;

//     Eigen::MatrixXf J;
//     Eigen::MatrixXf uv = Eigen::MatrixXf(2, NUMBER_OF_PIXELS);
//     Eigen::VectorXf depth = Eigen::VectorXf(NUMBER_OF_PIXELS);
//     Eigen::Matrix3f K = Eigen::Matrix3f::Identity();

//     depth << 31.15032, 32.37518, 31.58668;
//     K << 285.0, 0.0, 320.0,
//             0.0, 285.0, 240.0,
//             0.0, 0.0, 1.0;
//     uv << 324.47339, 324.02873, 316.40855,
//             256.61204, 235.51759, 269.08927;

//     Estimator::visjac_p(uv, depth, K, J);

//     auto Jtrue = Eigen::MatrixXf(6, 6);
//     Jtrue << -9.14918, 0.0, 0.14361, 0.26074, -285.07021, 16.61204,
//             0.0, -9.14918, 0.53329, 285.96828, -0.26074, -4.47339,
//             -8.80304, 0.0, 0.12444, -0.06336, -285.05695, -4.48241,
//             0.0, -8.80304, -0.13845, 285.0705, 0.06336, -4.02873,
//             -9.02279, 0.0, -0.1137, -0.36657, -285.04526, 29.08927,
//             0.0, -9.02279, 0.92093, 287.96907, 0.36657, 3.59145;

//     EXPECT_TRUE(J.isApprox(Jtrue, 1e-6));
// }

// // TEST(TestFlowVelocityParts, VelocityIsComputedCorrectly) {
// //     const int NUMBER_OF_PIXELS = 3;
// //     Eigen::MatrixXf J;
// //     Eigen::MatrixXf uv = Eigen::MatrixXf(2, NUMBER_OF_PIXELS);
// //     Eigen::VectorXf depth = Eigen::VectorXf(NUMBER_OF_PIXELS);
// //     Eigen::Matrix3f K = Eigen::Matrix3f::Identity();

// //     depth << 31.15032, 32.37518, 31.58668;
// //     K << 285.0, 0.0, 320.0,
// //             0.0, 285.0, 240.0,
// //             0.0, 0.0, 1.0;
// //     uv << 324.47339, 324.02873, 316.40855,
// //             256.61204, 235.51759, 269.08927;

// //     Estimator::visjac_p(uv, depth, K, J);

// //     Eigen::VectorXf flow(6);
// //     flow << 9.00558, 8.6159, 8.6786, 8.94149, 9.13649, 8.10185;
// //     Eigen::VectorXf vel;
// //     Estimator::compute_velocity(J, flow, vel);

// //     Eigen::VectorXf veltrue(6);
// //     veltrue << -1, -1, -1, 0.0, 0.0, 0.0;
// //     EXPECT_TRUE(vel.isApprox(veltrue, 1e-4));
// // }

// TEST(TestFlowVelocityParts, PixelDepthIsCorrectlyComputed) {
// //     Eigen::MatrixXf J;
// //     Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
// //     K << 285.0, 0.0, 320.0,
// //             0.0, 285.0, 240.0,
// //             0.0, 0.0, 1.0;
// //     Eigen::Matrix3f cam_R_enu = Eigen::Matrix3f::Identity();
// //     cam_R_enu << 1.0, 0.0, 0.0,
// //             0.0, 0.0, -1.0,
// //             0.0, 1.0, 0.0;
// //     float height = 20.0;
// //     Eigen::Vector2f pixel = Eigen::Vector2f(323.57283, 255.75045);

// //     float Z = Estimator::get_pixel_z_in_camera_frame(pixel, cam_R_enu, K, height);

// //     EXPECT_FLOAT_EQ(Z, -361.89433);
// }

// TEST(TestFlowVelocityParts, RansacIsAbleToFilterOutDominantFlow) {
//     // take a sample of flow
//     // add some outliers, mimicking water, vessel, etc.
//     // check if the dominant flow is still recovered
// }

// TEST(TestFlowVelocityParts, PureRotationGivesZeroVelocity) {
//     // take a sample of flow
//     // add some outliers, mimicking water, vessel, etc.
//     // check if the dominant flow is still recovered
// }

// TEST(TestFlowVelocityParts, FullVelocityOnRealData) {
//     auto estimator = Estimator();
//     std::string dir = TEST_DIR;

//     cv::Mat frame0;
//     frame0 = cv::imread(dir + "frame0.png", cv::IMREAD_COLOR);
//     cv::cvtColor(frame0, frame0, cv::COLOR_BGR2RGB);

//     cv::Mat frame1;
//     frame1 = cv::imread(dir + "frame1.png", cv::IMREAD_COLOR);
//     cv::cvtColor(frame1, frame1, cv::COLOR_BGR2RGB);

// //    cv::imshow("frame", frame0);
// //    cv::waitKey(0);
// //    cv::imshow("frame", frame1);
// //    cv::waitKey(0);

//     float t0 = 508.652;
//     float t1 = 508.685;
//     Eigen::Matrix3f cam_R_enu;
//     cam_R_enu << 0.499455, -0.624689, 0.600257,
//             -0.866339, -0.360991, 0.34517,
//             0.00106359, -0.692423, -0.721491;
//     float height = 6.30986;
//     Eigen::Vector3f r;
//     r << 0.127711, 0.00500937, -0.0735654;
//     Eigen::Vector3f cam_omega;
//     cam_omega << 0.09909591711111113, 0.025986013, -0.030931112611111106;
//     Eigen::Vector3f drone_omega;
//     drone_omega << -0.03847748836363636, -0.10718755563636365, 0.0076125204545454545;
//     Eigen::Vector3f gt_vel_ned = {1.282000065, 3.300000191, 0.229000017};

//     Eigen::Matrix3f K;
//     K << 385.402, 0.0, 322.133,
//             0.0, 384.882, 240.013,
//             0.0, 0.0, 1.0;

// //     // first call simply record the previous frame
// //     auto ret = estimator.update_flow_velocity(
// //             frame0, t0, cam_R_enu, r, K, height, cam_omega, drone_omega);
// //     EXPECT_TRUE(ret.isApprox(Eigen::Vector3f::Zero(), 1e-6));

// //     // calculate velocity
// //     ret = estimator.update_flow_velocity(
// //             frame1, t1, cam_R_enu, r, K, height, cam_omega, drone_omega);

// //     Eigen::Matrix3f ned_R_enub = Eigen::Matrix3f::Identity();
// //     ned_R_enub << 0.0, 1.0, 0.0,
// //             1.0, 0.0, 0.0,
// //             0.0, 0.0, -1.0;
// //     Eigen::Vector3f gt_vel_enu = ned_R_enub * gt_vel_ned;

// //     auto ret_xy = ret.segment(0, 2);
// //     auto gt_xy = gt_vel_enu.segment(0, 2);

// //     EXPECT_NEAR(ret_xy.norm(), gt_xy.norm(), 2e-1);

// //     EXPECT_TRUE(ret_xy.isApprox(gt_xy, 1e-1))
// //                         << "computed vel: " << std::endl << ret << std::endl
// //                         << "ground truth: " << std::endl << gt_vel_enu << std::endl;

// // Estimator::update_flow_velocity(frame, time, cam_R_enu, &r, &K, height, omega)
// //  to test this I'll need:
// //      1) X two frames
// //      2) X associated time stamps
// //      3) X cam_R_enu
// //      4) PJ averaged angular velocity from camera gyro
// //      5) X height
// //      6) X arm length r - it's fixed
// //      7) PJ ground truth velocity
// //      8) PJ averaged angular velocity of the drone
// // X - dump from the function
// // PJ - get from PJ

// }
