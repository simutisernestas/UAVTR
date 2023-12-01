/**
 * Kalman filter implementation using Eigen. Based on the following
 * introductory paper:
 *
 *     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
 *
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 */
#pragma once

#include <Eigen/Dense>

class KalmanFilter {

public:
    /**
     * Create a Kalman filter with the specified matrices.
     *   A - System dynamics matrix
     *   C - Output matrix
     *   Q - Process noise covariance
     *   R - Measurement noise covariance
     *   P - Estimate error covariance
     */
    KalmanFilter(
            const Eigen::MatrixXf &A,
            const Eigen::MatrixXf &C,
            const Eigen::MatrixXf &Q,
            const Eigen::MatrixXf &R,
            const Eigen::MatrixXf &P);

    [[nodiscard]] inline bool is_initialized() const { return initialized; }

    /**
     * Initialize the filter with a guess for initial states.
     */
    void init(const Eigen::VectorXf &x0);

    void update(const Eigen::VectorXf &y);

    // custom update
    void update(const Eigen::VectorXf &y,
                const Eigen::MatrixXf &C_cus,
                const Eigen::MatrixXf &R_cus);

    void predict(const Eigen::MatrixXf &A);

    /**
     * Return the current state and time.
     */
    Eigen::VectorXf state() { return x_hat; };

    Eigen::MatrixXf covariance() { return P; };

private:
    // Matrices for computation
    Eigen::MatrixXf A, B, C, Q, R, P, K, P0;

    // System dimensions
    int n;

    // Is the filter initialized?
    bool initialized;

    // n-size identity
    Eigen::MatrixXf I;

    // Estimated states
    Eigen::VectorXf x_hat;

    std::mutex mtx_;
};
