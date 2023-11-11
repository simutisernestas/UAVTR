#include "estimator.hpp"
#include <Eigen/Sparse>

Estimator::Estimator()
{
    const double dt = 1.0 / 128.0;
    Eigen::MatrixXd A(12, 12);
    A << 1, 0, 0, dt, 0, 0, -.5 * dt * dt, 0, 0, -.5 * dt * dt, 0, 0,
        0, 1, 0, 0, dt, 0, 0, -.5 * dt * dt, 0, 0, -.5 * dt * dt, 0,
        0, 0, 1, 0, 0, dt, 0, 0, -.5 * dt * dt, 0, 0, -.5 * dt * dt,
        0, 0, 0, 1, 0, 0, -dt, 0, 0, -dt, 0, 0,
        0, 0, 0, 0, 1, 0, 0, -dt, 0, 0, -dt, 0,
        0, 0, 0, 0, 0, 1, 0, 0, -dt, 0, 0, -dt,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

    Eigen::MatrixXd B(9, 3);
    B << -.5 * dt * dt, 0, 0,
        0, -.5 * dt * dt, 0,
        0, 0, -.5 * dt * dt,
        -dt, 0, 0,
        0, -dt, 0,
        0, 0, -dt,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0;

    Eigen::MatrixXd C(2, 12);
    C << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    Eigen::MatrixXd Q(12, 12);
    Q = Eigen::MatrixXd::Identity(12, 12) * .5;
    Q.block(9, 9, 3, 3) = Eigen::MatrixXd::Zero(3, 3);

    Eigen::MatrixXd R(2, 2);
    R = Eigen::MatrixXd::Identity(2, 2) * 2.5;

    Eigen::MatrixXd P(12, 12);
    P = Eigen::MatrixXd::Identity(12, 12) * 100.0;

    kf_ = std::make_unique<KalmanFilter>(dt, A, B, C, Q, R, P);

    const std::array<double, 3> a = {1.0, -1.56101808, 0.64135154};
    const std::array<double, 3> b = {0.02008337, 0.04016673, 0.02008337};
    // Create a low pass filter objects.
    for (int i = 0; i < 3; i++)
        lp_acc_filter_arr_[i] = std::make_unique<LowPassFilter<double, 3>>(b, a);

    std::cout << A << std::endl;
}

void get_A(Eigen::MatrixXd &A, double dt)
{
    A.setZero();
    A << 1, 0, 0, dt, 0, 0, -.5 * dt * dt, 0, 0, -.5 * dt * dt, 0, 0,
        0, 1, 0, 0, dt, 0, 0, -.5 * dt * dt, 0, 0, -.5 * dt * dt, 0,
        0, 0, 1, 0, 0, dt, 0, 0, -.5 * dt * dt, 0, 0, -.5 * dt * dt,
        0, 0, 0, 1, 0, 0, -dt, 0, 0, -dt, 0, 0,
        0, 0, 0, 0, 1, 0, 0, -dt, 0, 0, -dt, 0,
        0, 0, 0, 0, 0, 1, 0, 0, -dt, 0, 0, -dt,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
}

void get_B(Eigen::MatrixXd &B, double dt)
{
    B.setZero();
    B << -.5 * dt * dt, 0, 0,
        0, -.5 * dt * dt, 0,
        0, 0, -.5 * dt * dt,
        -dt, 0, 0,
        0, -dt, 0,
        0, 0, -dt,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0;
}

// TODO: rename
Eigen::Vector3d Estimator::compute_pixel_rel_position(
    const Eigen::Vector2d &bbox_c, const Eigen::Matrix3d &cam_R_enu,
    const Eigen::Matrix3d &K, const double height, bool update)
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
    if (!update)
        return Pt;

    if (kf_->is_initialized())
    {
        Eigen::Vector2d xy_meas(2);
        xy_meas << Pt[0], Pt[1];
        kf_->update(xy_meas);
    }
    else
    {
        Eigen::VectorXd x0(12);
        x0 << Pt[0], Pt[1], Pt[2], 0, 0, 0, 0, 0, 0, 0, 0, 0;
        kf_->init(0.005, x0);
    }

    std::cout << "covariance: " << std::endl
              << kf_->covariance() << std::endl
              << std::endl;
    std::cout << "state" << std::endl
              << kf_->state() << std::endl
              << std::endl;

    return Pt;
}

void Estimator::update_height(const double height)
{
    if (!kf_->is_initialized())
        return;

    static Eigen::MatrixXd C_height(1, 12);
    C_height << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    static Eigen::MatrixXd R(1, 1);
    R << 0.1; // height measurement noise

    Eigen::VectorXd h(1);
    h << -height;
    // the relative height is negative
    kf_->update(h, C_height, R);
}

void Estimator::update_imu_accel(const Eigen::Vector3d &accel, double dt)
{
    if (!kf_->is_initialized())
        return;

    // copy accel vector into eigen vector
    auto copy = accel;
    // filter accel
    for (int i = 0; i < 3; i++)
        copy[i] = lp_acc_filter_arr_[i]->filter(copy[i]);

    // update A and B matrices
    Eigen::MatrixXd A(12, 12);
    get_A(A, dt);

    // Eigen::MatrixXd B(9, 3);
    // get_B(B, dt);
    // kf_->predict(accel, A, B);

    kf_->predict(A);

    // update C matrix
    static Eigen::MatrixXd C_accel(3, 12);
    C_accel << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
    static Eigen::MatrixXd R_accel(3, 3);
    R_accel = Eigen::MatrixXd::Identity(3, 3) * 1.0;
    kf_->update(accel, C_accel, R_accel);
}

#if 1
Eigen::MatrixXd visjac_p(const Eigen::MatrixXd &uv,
                         const Eigen::VectorXd &depth,
                         const Eigen::Matrix3d &K)
{
    Eigen::MatrixXd L(0, 6);
    Eigen::Matrix3d Kinv = K.inverse();

    for (int i = 0; i < uv.cols(); i++)
    {
        double z = depth(i);
        Eigen::Vector3d p(uv(0, i), uv(1, i), 1.0);

        // convert to normalized image-plane coordinates
        Eigen::Vector3d xy = Kinv * p;
        double x = xy(0);
        double y = xy(1);

        // 2x6 Jacobian for this point
        Eigen::Matrix<double, 2, 6> Lp;
        Lp << -1 / z, 0, x / z, x * y, -(1 + x * x), y,
            0, -1 / z, y / z, 1 + y * y, -x * y, -x;
        Lp = K.block(0, 0, 2, 2) * Lp;

        // stack them vertically
        L.conservativeResize(L.rows() + 2, Eigen::NoChange);
        L.bottomRows(2) = Lp;
    }

    return L;
}

void Estimator::update_flow_velocity(cv::Mat &frame, const Eigen::Matrix3d &cam_R_enu,
                                     const Eigen::Matrix3d &K, const double height)
{
    using namespace cv;
    std::cout << "update flow velocity" << std::endl;

    if (!prev_frame_)
    {
        cvtColor(frame, frame, COLOR_RGB2GRAY);
        goodFeaturesToTrack(frame, p0_, 100, 0.3, 7, Mat(), 7, false, 0.04);
        prev_frame_ = std::make_unique<cv::Mat>(frame);
        return;
    }

    cvtColor(frame, frame, COLOR_RGB2GRAY);

    // if (p0_.size() < 30)
    goodFeaturesToTrack(frame, p0_, 100, 0.3, 7, Mat(), 7, false, 0.04);
    if (p0_.size() < 5)
    {
        *prev_frame_ = frame;
        std::cout << "Good old size" << p0_.size() << std::endl;
        return;
    }

    std::vector<uchar> status;
    std::vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(*prev_frame_, frame, p0_, p1_, status, err, Size(15, 15), 2, criteria);
    std::vector<Point2f> good_new;
    std::vector<Point2f> good_old;
    for (uint i = 0; i < p0_.size(); i++)
    {
        if (status[i] != 1)
            continue;
        good_new.push_back(p1_[i]);
        good_old.push_back(p0_[i]);
    }
    if (good_old.size() < 5)
    {
        *prev_frame_ = frame;
        std::cout << "Good old size" << good_old.size() << std::endl;
        return;
    }

    // depths
    Eigen::VectorXd depth(good_old.size());
    for (uint i = 0; i < good_old.size(); i++)
    {
        auto Pt = compute_pixel_rel_position(Eigen::Vector2d(good_old[i].x, good_old[i].y), cam_R_enu, K, height, false);
        depth(i) = Pt.norm();
        if (depth(i) > 100)
        {
            p0_.clear();
            *prev_frame_ = frame;
            return;
        }
    }
    auto uv = Eigen::MatrixXd(2, good_old.size());
    for (uint i = 0; i < good_old.size(); i++)
    {
        uv(0, i) = good_old[i].x;
        uv(1, i) = good_old[i].y;
    }
    auto J = visjac_p(uv, depth, K);
    // std::cout << "depths: " << std::endl
    //           << depth << std::endl
    //           << std::endl;
    // std::cout << J << std::endl;

    auto flow_vecs = std::vector<Eigen::Vector2d>();
    for (uint i = 0; i < good_old.size(); i++)
        flow_vecs.push_back(Eigen::Vector2d(good_new[i].x - good_old[i].x,
                                            good_new[i].y - good_old[i].y));
    Eigen::VectorXd flow_vecs_vec(2 * flow_vecs.size());
    for (uint i = 0; i < flow_vecs.size(); i++)
    {
        flow_vecs_vec(2 * i) = flow_vecs[i][0];
        flow_vecs_vec(2 * i + 1) = flow_vecs[i][1];
    }
    // std::cout << flow_vecs_vec << std::endl
    //           << std::endl;

    *prev_frame_ = frame;
    p0_ = good_new;

    // draw on a a copy frame
    Mat copy;
    frame.copyTo(copy);
    cvtColor(copy, copy, COLOR_GRAY2BGR);
    for (uint i = 0; i < p0_.size(); i++)
    {
        circle(copy, p0_[i], 3, Scalar(0, 255, 0), -1);
    }
    std::cout << p0_.size() << std::endl;
    imshow("flow", copy);
    waitKey(1);

    // solve NNLS, A - J (n,2), b - flow_vecs_vec (2n,1)
    Eigen::LeastSquaresConjugateGradient<Eigen::MatrixXd> lscg{};
    lscg.compute(J);
    std::cout << J.rows() << " " << J.cols() << std::endl;
    std::cout << flow_vecs_vec.rows() << " " << flow_vecs_vec.cols() << std::endl;
    auto x = lscg.solve(flow_vecs_vec);
    std::cout << "error: " << lscg.error() << std::endl;
    if (lscg.error() > 1.0) {
        throw std::runtime_error("lscg error too high");
    }
    // std::cout << flow_vecs_vec << std::endl;
    std::cout << "VELOCITY FROM FLOW " << (cam_R_enu * x.segment(0, 3)).transpose() << std::endl;

    if (!kf_->is_initialized())
        return;

    static Eigen::MatrixXd C_vel(2, 12);
    C_vel << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
    static Eigen::MatrixXd R_vel(2, 2);
    R_vel = Eigen::MatrixXd::Identity(2, 2) * 1.0;
    kf_->update(x.segment(0, 2), C_vel, R_vel);

    auto cov = kf_->covariance();
    // check for nans
    for (int i = 0; i < cov.rows(); i++)
        for (int j = 0; j < cov.cols(); j++)
            if (std::isnan(cov(i, j)))
                throw std::runtime_error("covariance is nan");
}
#endif