#include "estimator.hpp"

Estimator::Estimator()
{
    double dt = 0.005;

    Eigen::MatrixXd A(9, 9);
    A << 1, 0, 0, dt, 0, 0, -.5 * dt * dt, 0, 0,
        0, 1, 0, 0, dt, 0, 0, -.5 * dt * dt, 0,
        0, 0, 1, 0, 0, dt, 0, 0, -.5 * dt * dt,
        0, 0, 0, 1, 0, 0, -dt, 0, 0,
        0, 0, 0, 0, 1, 0, 0, -dt, 0,
        0, 0, 0, 0, 0, 1, 0, 0, -dt,
        0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1;
    std::cout << A << std::endl;

    Eigen::MatrixXd B(9, 3);
    B << 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        dt, 0, 0,
        0, dt, 0,
        0, 0, dt,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0;

    Eigen::MatrixXd C(3, 9);
    C << 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, -0,
        0, 0, 1, 0, 0, 0, 0, 0, 0;

    Eigen::MatrixXd Q(9, 9);
    Q = Eigen::MatrixXd::Identity(9, 9) * 0.05;
    Q(6, 6) = 0;
    Q(7, 7) = 0;
    Q(8, 8) = 0;
    Eigen::MatrixXd R(3, 3);
    R = Eigen::MatrixXd::Identity(3, 3) * 100;
    R(3, 3) = .1;
    Eigen::MatrixXd P(9, 9);
    P = Eigen::MatrixXd::Identity(9, 9) * 1000;

    kf_ = std::make_unique<KalmanFilter>(dt, A, B, C, Q, R, P);
}

// TODO: rename
Eigen::Vector3d Estimator::compute_pixel_rel_position(
    const Eigen::Vector2d &bbox_c, const Eigen::Matrix3d &cam_R_enu,
    const Eigen::Matrix3d &K, const double height)
{
    Eigen::Matrix<double, 3, 3> Kinv = K.inverse();
    Eigen::Vector3d lr;
    lr << 0, 0, -1;
    Eigen::Vector3d Puv_hom;
    Puv_hom << bbox_c[0], bbox_c[1], 1;
    Eigen::Vector3d Pc = Kinv * Puv_hom;
    Eigen::Vector3d ls = cam_R_enu * (Pc / Pc.norm());
    double d = height / (lr.transpose() * ls);
    Eigen::Vector3d Pt = ls * d;
    if (kf_->is_initialized())
        kf_->update(Pt);
    else
    {
        Eigen::VectorXd x0(9);
        x0 << Pt[0], Pt[1], Pt[2], 0, 0, 0, 0, 0, 0;
        kf_->init(0.005, x0);
    }
    auto x_hat = kf_->state();
    std::cout << x_hat << std::endl;
    Pt << x_hat[0], x_hat[1], x_hat[2];
    return Pt;
}

void Estimator::update_flow_velocity(const cv::Mat &frame)
{
    using namespace cv;

    if (!prev_frame_)
    {
        cvtColor(frame, frame, COLOR_RGB2GRAY);
        goodFeaturesToTrack(frame, p0_, 100, 0.3, 7, Mat(), 7, false, 0.04);
        prev_frame_ = std::make_unique<cv::Mat>(frame);
        return;
    }

    cvtColor(frame, frame, COLOR_RGB2GRAY);
    std::vector<uchar> status;
    std::vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(*prev_frame_, frame, p0_, p1_, status, err, Size(15, 15), 2, criteria);
    std::vector<Point2f> good_new;
    for (uint i = 0; i < p0_.size(); i++)
    {
        if (status[i] == 1)
            good_new.push_back(p1_[i]);
    }
    *prev_frame_ = frame;
    p0_ = good_new;
}

void Estimator::update_imu_accel(const Eigen::Vector3d &accel)
{
    if (!kf_->is_initialized())
        return;

    kf_->predict(accel);
}
