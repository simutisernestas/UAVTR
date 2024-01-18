/**
 * Modified version of:
 * 
 * Implementation of KalmanFilter class.
 *
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 * 
 */

#include <mutex>
#include <stdexcept>

#include "kalman.hpp"

KalmanFilter::KalmanFilter(
    const Eigen::MatrixXf &A,
    const Eigen::MatrixXf &C,
    const Eigen::MatrixXf &Q,
    const Eigen::MatrixXf &R,
    const Eigen::MatrixXf &P)
    : A(A), C(C), Q(Q), R(R), P0(P),
      n(A.rows()), initialized(false),
      I(n, n), x_hat(n) {
  I.setIdentity();
  x_hat.setZero();
}

void KalmanFilter::init(const Eigen::VectorXf &x0) {
  // scoped lock
  std::scoped_lock lock(mtx_);
  x_hat = x0;
  P = P0;
  initialized.store(true);
}

void KalmanFilter::predict(const Eigen::MatrixXf &A_cus) {
  std::scoped_lock lock(mtx_);
  if (!initialized.load())
    throw std::runtime_error("Filter is not initialized!");

  x_hat = A_cus * x_hat;
  P = A_cus * P * A_cus.transpose() + Q;
}

void KalmanFilter::update(const Eigen::VectorXf &y) {
  std::scoped_lock lock(mtx_);
  if (!initialized.load())
    throw std::runtime_error("Filter is not initialized!");

  K = P * C.transpose() * (C * P * C.transpose() + R).inverse();
  x_hat += K * (y - C * x_hat);
  Eigen::MatrixXf IKC = (I - K * C);
  P = IKC * P * IKC.transpose() + K * R * K.transpose();
}

void KalmanFilter::update(const Eigen::VectorXf &y,
                          const Eigen::MatrixXf &C_cus,
                          const Eigen::MatrixXf &R_cus) {
  std::scoped_lock lock(mtx_);
  if (!initialized.load())
    throw std::runtime_error("Filter is not initialized!");

  K = P * C_cus.transpose() * (C_cus * P * C_cus.transpose() + R_cus).inverse();
  x_hat += K * (y - C_cus * x_hat);
  Eigen::MatrixXf IKC = (I - K * C_cus);
  P = IKC * P * IKC.transpose() + K * R_cus * K.transpose();
}

Eigen::VectorXf KalmanFilter::state() {
  if (!initialized.load())
    return Eigen::VectorXf::Zero(n);
  std::scoped_lock lock(mtx_);
  return x_hat;
}

Eigen::MatrixXf KalmanFilter::covariance() {
  if (!initialized.load())
    return Eigen::MatrixXf::Zero(n, n);
  std::scoped_lock lock(mtx_);
  return P;
}

void KalmanFilter::reset_boat_velocity() {
  std::scoped_lock lock(mtx_);
  x_hat[6] = 0;
  x_hat[7] = 0;
  P(6, 6) = 1;
  P(7, 7) = 1;
}
