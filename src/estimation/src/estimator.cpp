#include "estimator.hpp"
#include <Eigen/Sparse>
#include <numeric>
#include <random>

Estimator::Estimator() {
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
    // Q = Eigen::MatrixXd::Identity(12, 12) * .5;
    Q.block(9, 9, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
    Eigen::MatrixXd Q99(9, 9);
    Q99
            << 9.2297163e-06, 1.7489764e-06, 9.8769633e-07, 9.8379522e-08, -8.3961406e-09, -7.7924722e-08, -5.1577410e-06, 4.8649296e-06, 1.2655392e-05,
            1.7489764e-06, 3.6980000e-06, 7.7610747e-07, 7.2662781e-08, 2.1758865e-08, -1.2657726e-08, -1.1019557e-05, 6.1990057e-06, 3.6906130e-06,
            9.8769633e-07, 7.7610747e-07, 3.0083441e-06, 9.4196289e-08, 4.9950499e-08, 1.9305785e-08, -8.7744011e-06, -1.5469293e-06, -2.6607292e-07,
            9.8379522e-08, 7.2662781e-08, 9.4196289e-08, 4.4736172e-06, 1.2455506e-06, 2.3520657e-07, -5.0299648e-04, -1.8283247e-04, -3.3572126e-05,
            -8.3961406e-09, 2.1758865e-08, 4.9950499e-08, 1.2455506e-06, 3.4215782e-06, 4.6726025e-07, -1.8273460e-04, -3.3796436e-04, -6.9801495e-05,
            -7.7924722e-08, -1.2657726e-08, 1.9305785e-08, 2.3520657e-07, 4.6726025e-07, 2.8831977e-06, -3.4125542e-05, -7.0074971e-05, -2.7096335e-04,
            -5.1577410e-06, -1.1019557e-05, -8.7744011e-06, -5.0299648e-04, -1.8273460e-04, -3.4125542e-05, 1.2915180e-01, 4.6890670e-02, 8.6542333e-03,
            4.8649296e-06, 6.1990057e-06, -1.5469293e-06, -1.8283247e-04, -3.3796436e-04, -7.0074971e-05, 4.6890670e-02, 8.6584708e-02, 1.8099087e-02,
            1.2655392e-05, 3.6906130e-06, -2.6607292e-07, -3.3572126e-05, -6.9801495e-05, -2.7096335e-04, 8.6542333e-03, 1.8099087e-02, 6.9678758e-02;
    Q.block(0, 0, 9, 9) = Q99;

    // R = np.array([[  1.0550187e+01,3.4368357e+00,2.7349897e+00,1.0717672e-01,2.8714117e-01,-8.2645767e-03,],
    // [3.4368357e+00,2.6576614e+00,1.3028051e+00,5.8013531e-02,2.4486553e-01,2.4525120e-02,],
    // [2.7349897e+00,1.3028051e+00,1.5327067e+00,1.1493263e-01,3.8987427e-02,3.1687897e-02,],
    // [1.0717672e-01,5.8013531e-02,1.1493263e-01,2.0988398e-01,2.0624759e-02,1.2513926e-02,],
    // [2.8714117e-01,2.4486553e-01,3.8987427e-02,2.0624759e-02,2.2076985e-01,4.5288393e-02,],
    // [-8.2645767e-03,2.4525120e-02,3.1687897e-02,1.2513926e-02,4.5288393e-02,1.9426574e-01,],
    // ])

    Eigen::MatrixXd R(2, 2);
    R << 1.0550187e+01, 3.4368357e+00,
            3.4368357e+00, 2.6576614e+00;

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

void get_A(Eigen::MatrixXd &A, double dt) {
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

void get_B(Eigen::MatrixXd &B, double dt) {
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
        const Eigen::Matrix3d &K, const double height, bool update) {
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

    if (kf_->is_initialized()) {
        Eigen::Vector2d xy_meas(2);
        xy_meas << Pt[0], Pt[1];
        kf_->update(xy_meas);
    } else {
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

void Estimator::update_height(const double height) {
    if (!kf_->is_initialized())
        return;

    static Eigen::MatrixXd C_height(1, 12);
    C_height << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    static Eigen::MatrixXd R(1, 1);
    R << 1.5327067e+00; // height measurement noise

    Eigen::VectorXd h(1);
    h << -height;
    // the relative height is negative
    kf_->update(h, C_height, R);
}

void Estimator::update_imu_accel(const Eigen::Vector3d &accel, double dt) {
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
    R_accel << 2.0988398e-01, 2.0624759e-02, 1.2513926e-02,
            2.0624759e-02, 2.2076985e-01, 4.5288393e-02,
            1.2513926e-02, 4.5288393e-02, 1.9426574e-01;

    kf_->update(accel, C_accel, R_accel);
}

#if 1

void visjac_p(const Eigen::MatrixXd &uv,
              const Eigen::VectorXd &depth,
              const Eigen::Matrix3d &K,
              Eigen::MatrixXd &L) {
    L.resize(depth.size() * 2, 6);
    L.setZero();
    Eigen::Matrix3d Kinv = K.inverse();

    for (int i = 0; i < uv.cols(); i++) {
        double z = depth(i);
        Eigen::Vector3d p(uv(0, i), uv(1, i), 1.0);

        // convert to normalized image-plane coordinates
        Eigen::Vector3d xy = Kinv * p;
        double x = xy(0);
        double y = xy(1);

        // 2x6 Jacobian for this point
        Eigen::Matrix<double, 2, 6> Lp;
        Lp << -1 / z, 0.0, x / z, x * y, -(1 + x * x), y,
                0.0, -1 / z, y / z, 1 + y * y, -x * y, -x;
        Lp = K.block(0, 0, 2, 2) * Lp;

        // push into Jacobian
        L.block(2 * i, 0, 2, 6) = Lp;
    }
}

void RANSAC_vel_regression(const Eigen::MatrixXd &J,
                           const Eigen::VectorXd &flow_vectors,
                           Eigen::VectorXd &cam_vel_est) {
    // https://rpg.ifi.uzh.ch/docs/Visual_Odometry_Tutorial.pdf slide 68
    // >> outlier_percentage = .75
    // >>> np.log(1 - 0.999) / np.log(1 - (1 - outlier_percentage) ** n_samples)
    // 438.63339476983924
    size_t n_iterations = 439;
    size_t n_samples{3}; // minimum required to fit model
    size_t n_points = flow_vectors.rows() / 2;

    std::random_device rd;                                  // obtain a random number from hardware
    std::mt19937 gen(rd());                                 // seed the generator
    std::uniform_int_distribution<> distr(0, n_points - 1); // define the range

    auto best_inliers = std::vector<size_t>{}; // best inlier indices
    auto min_error = std::numeric_limits<double>::max();

    Eigen::MatrixXd J_samples(n_samples * 2, J.cols());
    J_samples.setZero();
    Eigen::VectorXd flow_samples(n_samples * 2);
    flow_samples.setZero();

    auto solve_sampled = [&J, &flow_vectors, &J_samples, &flow_samples](
            const std::vector<size_t> &sample_idxs, Eigen::VectorXd &sol) {
        // take sampled data
        for (size_t i{0}; i < sample_idxs.size(); ++i) {
            J_samples.block(i * 2, 0, 2, J.cols()) = J.block(sample_idxs[i] * 2, 0, 2, J.cols());
            flow_samples.segment(i * 2, 2) = flow_vectors.segment(sample_idxs[i] * 2, 2);
        }
        // solve for velocity
        sol = (J_samples.transpose() * J_samples).ldlt().solve(J_samples.transpose() * flow_samples);
    };

    Eigen::VectorXd x_est(J.cols());
    for (size_t iter{0}; iter <= n_iterations; ++iter) {
        // randomly select n_samples from data
        std::vector<size_t> sample_idxs(n_samples);
        for (size_t i{0}; i < n_samples; ++i)
            sample_idxs[i] = distr(gen);

        // solve
        solve_sampled(sample_idxs, x_est);

        // compute error
        // std::cout << "x_est: " << x_est << std::endl;
        // std::cout << "J: " << J.cols() << std::endl;
//        assert(J.rows() == flow_vectors.rows());
//        assert(J.cols() == x_est.rows());
        Eigen::VectorXd error = J * x_est - flow_vectors;

        // compute inliers
        std::vector<size_t> inlier_idxs;
        double error_sum{0};
        for (size_t i{0}; i < n_points; ++i) {
            auto error_x = error(i * 2);
            auto error_y = error(i * 2 + 1);
            double error_norm = std::sqrt(error_x * error_x + error_y * error_y);
            error_sum += error_norm;
            if (error_norm < 10) // TODO: this is pixels, with 10 get anywhere from 50 to 4000 inliers
                inlier_idxs.push_back(i);
        }
        error_sum /= (double) n_points;
        std::cout << "Error avg:" << error_sum << std::endl;
        std::cout << "Inliers:" << inlier_idxs.size() << std::endl;
//        const double allowed_error_perc = 0.1;
//        const double allowed_error = allowed_error_perc * error_sum;
//        for (size_t i{0}; i < n_points; ++i) {
//            auto error_x = error(i * 2);
//            auto error_y = error(i * 2 + 1);
//            double error_norm = std::sqrt(error_x * error_x + error_y * error_y);
//
//        }

        if (error_sum < min_error) {
            best_inliers = inlier_idxs;
            min_error = error_sum;
        }

        // TODO: early stopping ?
    }

    std::cout << "Return:" << std::endl;
    J_samples.resize(best_inliers.size() * 2, J.cols());
    flow_samples.resize(best_inliers.size() * 2);
    solve_sampled(best_inliers, cam_vel_est);
    std::cout << "Best inliers: " << cam_vel_est << std::endl;
}

void Estimator::update_flow_velocity(cv::Mat &frame, const Eigen::Matrix3d &cam_R_enu,
                                     const Eigen::Matrix3d &K, const double height) {
    if (!prev_frame_) {
        prev_frame_ = std::make_unique<cv::Mat>(frame);
        return;
    }

    cv::Mat flow;
    optflow->calc(*prev_frame_, frame, flow);
    // here depending on which frame is used later the sign could be fliped ?
    // also depths could be computed wrong ?

    for (int y = 0; y < frame.rows; y += 16) {
        for (int x = 0; x < frame.cols; x += 16) {
            // Get the flow from `flow`, which is a 2-channel matrix
            const cv::Point2f &fxy = flow.at<cv::Point2f>(y, x);

            // Draw lines on `frame` to represent flow
            cv::line(frame, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), cv::Scalar(0, 255, 0));
            cv::circle(frame, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
        }
    }

    // extract the dominant flow direction
    std::vector<cv::Point2f> flow_vecs;
    flow_vecs.reserve(frame.rows * frame.cols / 64);
    std::vector<cv::Point> samples;
    samples.reserve(frame.rows * frame.cols / 64);
    // 4800 data points each iteration
    for (int row = 0; row < frame.rows; row += 8) {
        for (int col = 0; col < frame.cols; col += 8) {
            // Get the flow from `flow`, which is a 2-channel matrix
            const cv::Point2f &fxy = flow.at<cv::Point2f>(row, col);
            flow_vecs.push_back(fxy);
            samples.push_back(cv::Point(row, col));
        }
    }
    auto dominant_flow_vec = std::accumulate(flow_vecs.begin(), flow_vecs.end(), cv::Point2f(0, 0));
    dominant_flow_vec.x /= flow_vecs.size();
    dominant_flow_vec.y /= flow_vecs.size();

    // show the dominant flow vector on the image
    cv::line(frame, cv::Point(frame.cols / 2, frame.rows / 2),
             cv::Point(cvRound(frame.cols / 2 + dominant_flow_vec.x), cvRound(frame.rows / 2 + dominant_flow_vec.y)),
             cv::Scalar(0, 0, 255), 5);

    Eigen::VectorXd depth(samples.size());
    Eigen::MatrixXd uv = Eigen::MatrixXd(2, samples.size());
    Eigen::VectorXd flow_eigen(2 * flow_vecs.size());
    for (size_t i = 0; i < samples.size(); i++) {
        // fill in pixel depths
        const Eigen::Vector3d Pt = compute_pixel_rel_position(
                Eigen::Vector2d(samples[i].x, samples[i].y), cam_R_enu, K, height, false);
        const double pixel_depth = Pt.norm();
        if (pixel_depth > 100) {
            continue;
//            std::cout << "Coords: " << samples[i].x << " " << samples[i].y
//                      << " Depth: " << pixel_depth << std::endl;
//            throw std::runtime_error("pixel depth too high");
        }
        depth(i) = pixel_depth;

        // store the uv coordinates
        uv(0, i) = static_cast<double>(samples[i].x);
        uv(1, i) = static_cast<double>(samples[i].y);

        flow_eigen(2 * i) = flow_vecs[i].x;
        flow_eigen(2 * i + 1) = flow_vecs[i].y;
    }
    Eigen::MatrixXd J; // Jacobian
    visjac_p(uv, depth, K, J);

    Eigen::VectorXd cam_vel_est;
    RANSAC_vel_regression(J, flow_eigen, cam_vel_est);

    // Display the image with vectors
    cv::imshow("Optical Flow Vectors", frame);
    cv::waitKey(1);
    *prev_frame_ = frame;

    return;

    // using namespace cv;
    // std::cout << "update flow velocity" << std::endl;

    // if (!prev_frame_)
    // {
    //     cvtColor(frame, frame, COLOR_RGB2GRAY);
    //     goodFeaturesToTrack(frame, p0_, 100, 0.3, 7, Mat(), 7, false, 0.04);
    //     prev_frame_ = std::make_unique<cv::Mat>(frame);
    //     return;
    // }

    // cvtColor(frame, frame, COLOR_RGB2GRAY);

    // // if (p0_.size() < 30)
    // goodFeaturesToTrack(frame, p0_, 100, 0.3, 7, Mat(), 7, false, 0.04);
    // if (p0_.size() < 5)
    // {
    //     *prev_frame_ = frame;
    //     std::cout << "Good old size" << p0_.size() << std::endl;
    //     return;
    // }

    // std::vector<uchar> status;
    // std::vector<float> err;
    // TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    // calcOpticalFlowPyrLK(*prev_frame_, frame, p0_, p1_, status, err, Size(15, 15), 2, criteria);
    // std::vector<Point2f> good_new;
    // std::vector<Point2f> good_old;
    // for (uint i = 0; i < p0_.size(); i++)
    // {
    //     if (status[i] != 1)
    //         continue;
    //     good_new.push_back(p1_[i]);
    //     good_old.push_back(p0_[i]);
    // }
    // if (good_old.size() < 5)
    // {
    //     *prev_frame_ = frame;
    //     std::cout << "Good old size" << good_old.size() << std::endl;
    //     return;
    // }

    // // depths
    // Eigen::VectorXd depth(good_old.size());
    // for (uint i = 0; i < good_old.size(); i++)
    // {
    //     auto Pt = compute_pixel_rel_position(Eigen::Vector2d(good_old[i].x, good_old[i].y), cam_R_enu, K, height, false);
    //     depth(i) = Pt.norm();
    //     if (depth(i) > 100)
    //     {
    //         p0_.clear();
    //         *prev_frame_ = frame;
    //         return;
    //     }
    // }
    // auto uv = Eigen::MatrixXd(2, good_old.size());
    // for (uint i = 0; i < good_old.size(); i++)
    // {
    //     uv(0, i) = good_old[i].x;
    //     uv(1, i) = good_old[i].y;
    // }
    // auto J = visjac_p(uv, depth, K);
    // // std::cout << "depths: " << std::endl
    // //           << depth << std::endl
    // //           << std::endl;
    // // std::cout << J << std::endl;

    // auto flow_vecs = std::vector<Eigen::Vector2d>();
    // for (uint i = 0; i < good_old.size(); i++)
    //     flow_vecs.push_back(Eigen::Vector2d(good_new[i].x - good_old[i].x,
    //                                         good_new[i].y - good_old[i].y));
    // Eigen::VectorXd flow_vecs_vec(2 * flow_vecs.size());
    // for (uint i = 0; i < flow_vecs.size(); i++)
    // {
    //     flow_vecs_vec(2 * i) = flow_vecs[i][0];
    //     flow_vecs_vec(2 * i + 1) = flow_vecs[i][1];
    // }
    // // std::cout << flow_vecs_vec << std::endl
    // //           << std::endl;

    // *prev_frame_ = frame;
    // p0_ = good_new;

    // // draw on a a copy frame
    // Mat copy;
    // frame.copyTo(copy);
    // cvtColor(copy, copy, COLOR_GRAY2BGR);
    // for (uint i = 0; i < p0_.size(); i++)
    // {
    //     circle(copy, p0_[i], 3, Scalar(0, 255, 0), -1);
    // }
    // std::cout << p0_.size() << std::endl;
    // imshow("flow", copy);
    // waitKey(1);

    // // solve NNLS, A - J (n,2), b - flow_vecs_vec (2n,1)
    // Eigen::LeastSquaresConjugateGradient<Eigen::MatrixXd> lscg{};
    // lscg.compute(J);
    // std::cout << J.rows() << " " << J.cols() << std::endl;
    // std::cout << flow_vecs_vec.rows() << " " << flow_vecs_vec.cols() << std::endl;
    // auto x = lscg.solve(flow_vecs_vec);
    // std::cout << "error: " << lscg.error() << std::endl;
    // if (lscg.error() > 1.0)
    // {
    //     throw std::runtime_error("lscg error too high");
    // }
    // // std::cout << flow_vecs_vec << std::endl;
    // std::cout << "VELOCITY FROM FLOW " << (cam_R_enu * x.segment(0, 3)).transpose() << std::endl;

    // if (!kf_->is_initialized())
    //     return;

    // static Eigen::MatrixXd C_vel(2, 12);
    // C_vel << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
    // static Eigen::MatrixXd R_vel(2, 2);
    // R_vel = Eigen::MatrixXd::Identity(2, 2) * 1.0;
    // kf_->update(x.segment(0, 2), C_vel, R_vel);

    // auto cov = kf_->covariance();
    // // check for nans
    // for (int i = 0; i < cov.rows(); i++)
    //     for (int j = 0; j < cov.cols(); j++)
    //         if (std::isnan(cov(i, j)))
    //             throw std::runtime_error("covariance is nan");
}

#endif