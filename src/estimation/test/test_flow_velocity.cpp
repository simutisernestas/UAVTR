#include <eigen3/Eigen/Core>
#include "gtest/gtest.h"
#include "../include/estimator.hpp"

TEST(TestFlowVelocityParts, ImageJacobianGivesCorrectResult) {
    const int NUMBER_OF_PIXELS = 3;

    Eigen::MatrixXd J;
    Eigen::MatrixXd uv = Eigen::MatrixXd(2, NUMBER_OF_PIXELS);
    Eigen::VectorXd depth = Eigen::VectorXd(NUMBER_OF_PIXELS);
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();

    depth << 31.15032, 32.37518, 31.58668;
    K << 285.0, 0.0, 320.0,
            0.0, 285.0, 240.0,
            0.0, 0.0, 1.0;
    uv << 324.47339, 324.02873, 316.40855,
            256.61204, 235.51759, 269.08927;

    Estimator::visjac_p(uv, depth, K, J);

    auto Jtrue = Eigen::MatrixXd(6, 6);
    Jtrue << -9.14918, 0.0, 0.14361, 0.26074, -285.07021, 16.61204,
            0.0, -9.14918, 0.53329, 285.96828, -0.26074, -4.47339,
            -8.80304, 0.0, 0.12444, -0.06336, -285.05695, -4.48241,
            0.0, -8.80304, -0.13845, 285.0705, 0.06336, -4.02873,
            -9.02279, 0.0, -0.1137, -0.36657, -285.04526, 29.08927,
            0.0, -9.02279, 0.92093, 287.96907, 0.36657, 3.59145;

    EXPECT_TRUE(J.isApprox(Jtrue, 1e-6));
}

TEST(TestFlowVelocityParts, VelocityIsComputedCorrectly) {
    const int NUMBER_OF_PIXELS = 3;
    Eigen::MatrixXd J;
    Eigen::MatrixXd uv = Eigen::MatrixXd(2, NUMBER_OF_PIXELS);
    Eigen::VectorXd depth = Eigen::VectorXd(NUMBER_OF_PIXELS);
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();

    depth << 31.15032, 32.37518, 31.58668;
    K << 285.0, 0.0, 320.0,
            0.0, 285.0, 240.0,
            0.0, 0.0, 1.0;
    uv << 324.47339, 324.02873, 316.40855,
            256.61204, 235.51759, 269.08927;

    Estimator::visjac_p(uv, depth, K, J);

    Eigen::VectorXd flow(6);
    flow << 9.00558, 8.6159, 8.6786, 8.94149, 9.13649, 8.10185;
    Eigen::VectorXd vel;
    Estimator::compute_velocity(J, flow, vel);

    Eigen::VectorXd veltrue(6);
    veltrue << -1, -1, -1, 0.0, 0.0, 0.0;
    EXPECT_TRUE(vel.isApprox(veltrue, 1e-4));
}
