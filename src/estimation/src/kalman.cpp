/**
 * Implementation of KalmanFilter class.
 *
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 */

#include <stdexcept>
#include <mutex>

#include "kalman.hpp"

KalmanFilter::KalmanFilter(
        const Eigen::MatrixXd &A,
        const Eigen::MatrixXd &C,
        const Eigen::MatrixXd &Q,
        const Eigen::MatrixXd &R,
        const Eigen::MatrixXd &P)
        : A(A), C(C), Q(Q), R(R), P0(P),
          n(A.rows()), initialized(false),
          I(n, n), x_hat(n) {
    I.setIdentity();
}

void KalmanFilter::init(const Eigen::VectorXd &x0) {
    // scoped lock
    std::scoped_lock lock(mtx_);
    x_hat = x0;
    P = P0;
    initialized = true;
}

void KalmanFilter::predict(const Eigen::MatrixXd &A_cus) {
    std::scoped_lock lock(mtx_);
    if (!initialized)
        throw std::runtime_error("Filter is not initialized!");

    x_hat = A_cus * x_hat;
    P = A_cus * P * A_cus.transpose() + Q;
}

void KalmanFilter::update(const Eigen::VectorXd &y) {
    std::scoped_lock lock(mtx_);
    if (!initialized)
        throw std::runtime_error("Filter is not initialized!");

    K = P * C.transpose() * (C * P * C.transpose() + R).inverse();
    x_hat += K * (y - C * x_hat);
    Eigen::MatrixXd IKC = (I - K * C);
    P = IKC * P * IKC.transpose() + K * R * K.transpose();
}

void KalmanFilter::update(const Eigen::VectorXd &y,
                          const Eigen::MatrixXd &C_cus,
                          const Eigen::MatrixXd &R_cus) {
    std::scoped_lock lock(mtx_);
    if (!initialized)
        throw std::runtime_error("Filter is not initialized!");

    K = P * C_cus.transpose() * (C_cus * P * C_cus.transpose() + R_cus).inverse();
    x_hat += K * (y - C_cus * x_hat);
    Eigen::MatrixXd IKC = (I - K * C_cus);
    P = IKC * P * IKC.transpose() + K * R_cus * K.transpose();
}

