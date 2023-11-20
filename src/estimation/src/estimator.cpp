#include "estimator.hpp"
#include <Eigen/Sparse>
#include <numeric>
#include <random>

Estimator::Estimator() {
    const double dt = 1.0 / 128.0;
    Eigen::MatrixXd A(12, 12);
    get_A(A, dt);

    Eigen::MatrixXd C(2, 12);
    C << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    Eigen::MatrixXd Q(12, 12);
    Q.block(9, 9, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
    Eigen::MatrixXd Q99(9, 9);
    Q99
            << 1.4327232e-05, 4.5542407e-05, -3.6177772e-05, 3.8054527e-05, -3.7679753e-05, 3.7981345e-05, -3.6543158e-05, 6.6228309e-05, -1.5780837e-05,
            4.5542407e-05, 1.5206209e-04, -8.7649881e-05, 1.4784123e-04, -9.2957250e-05, 1.4787818e-04, -1.6386951e-04, 1.9778008e-04, -8.1902863e-05,
            -3.6177772e-05, -8.7649881e-05, 1.9678981e-04, -1.2330969e-05, 1.7859111e-04, 2.2915255e-05, -7.7550530e-05, -1.9246018e-04, -6.0079266e-05,
            3.8054527e-05, 1.4784123e-04, -1.2330969e-05, 4.0092404e-04, 4.2468562e-05, 2.0720649e-04, 1.9879875e-04, 3.7377742e-04, -3.4079141e-04,
            -3.7679753e-05, -9.2957250e-05, 1.7859111e-04, 4.2468562e-05, 3.8693461e-04, -2.2824766e-04, -5.0098784e-04, -1.2712694e-04, -2.5829339e-05,
            3.7981345e-05, 1.4787818e-04, 2.2915255e-05, 2.0720649e-04, -2.2824766e-04, 5.3843447e-04, 3.5249679e-04, 6.1598909e-05, -7.0989167e-05,
            -3.6543158e-05, -1.6386951e-04, -7.7550530e-05, 1.9879875e-04, -5.0098784e-04, 3.5249679e-04, 1.5456325e-01, 6.5116601e-02, 1.1024973e-02,
            6.6228309e-05, 1.9778008e-04, -1.9246018e-04, 3.7377742e-04, -1.2712694e-04, 6.1598909e-05, 6.5116601e-02, 1.2050196e-01, 1.9366697e-02,
            -1.5780837e-05, -8.1902863e-05, -6.0079266e-05, -3.4079141e-04, -2.5829339e-05, -7.0989167e-05, 1.1024973e-02, 1.9366697e-02, 6.7944942e-02;

    Q.block(0, 0, 9, 9) = Q99;

    Eigen::MatrixXd R(2, 2);
    R << 1.7650208e+01, 1.2699096e+01,
            1.2699096e+01, 1.5283947e+01;

    Eigen::MatrixXd P(12, 12);
    P = Eigen::MatrixXd::Identity(12, 12) * 10.0;

    kf_ = std::make_unique<KalmanFilter>(A, C, Q, R, P);

    const std::array<double, 3> a = {1.0, -1.56101808, 0.64135154};
    const std::array<double, 3> b = {0.02008337, 0.04016673, 0.02008337};
    // Create a low pass filter objects.
    for (int i = 0; i < 3; i++)
        lp_acc_filter_arr_[i] = std::make_unique<LowPassFilter<double, 3>>(b, a);

    std::cout << A << std::endl;

    optflow_->setGridStep({16, 16}); // increasing this reduces runtime
}

void Estimator::get_A(Eigen::MatrixXd &A, double dt) {
    A.setZero();
    // incorporate IMU after tests
    double ddt2 = dt * dt * .5 * 0.0;
    A << 1, 0, 0, dt, 0, 0, ddt2, 0, 0, 0, 0, 0,
            0, 1, 0, 0, dt, 0, 0, ddt2, 0, 0, 0, 0,
            0, 0, 1, 0, 0, dt, 0, 0, ddt2, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
}

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
        kf_->init(x0);
    }

//    std::cout << "covariance: " << std::endl
//              << kf_->covariance() << std::endl
//              << std::endl;
//    std::cout << "state" << std::endl
//              << kf_->state() << std::endl
//              << std::endl;

    return Pt;
}

void Estimator::update_height(const double height) {
    if (!kf_->is_initialized())
        return;

    static Eigen::MatrixXd C_height(1, 12);
    C_height << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    static Eigen::MatrixXd R(1, 1);
    R << 2.3927872e+00; // height measurement noise

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

    kf_->predict(A);

    static Eigen::MatrixXd C_accel(3, 12);
    C_accel << 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0;
    static Eigen::MatrixXd R_accel(3, 3);
    R_accel << 4.3669211e+00, 3.1101683e-01, -7.3201366e-02,
            3.1101683e-01, 4.1195669e+00, -3.5575474e-01,
            7.3201366e-02, -3.5575474e-01, 1.9104830e+00;

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
    const size_t n_iterations = 439;
    const size_t n_samples{3}; // minimum required to fit model
    const size_t n_points = flow_vectors.rows() / 2;

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

    // compute median absolute deviation (MAD) for the target flow vectors, y = flow_vectors
    double residual_threshold{};
    {
        std::vector<double> abs_flow;
        abs_flow.reserve(n_points);
        for (size_t i{0}; i < n_points; ++i) {
            auto error_x = flow_vectors(i * 2);
            auto error_y = flow_vectors(i * 2 + 1);
            double error_norm = std::sqrt(error_x * error_x + error_y * error_y);
            abs_flow.push_back(error_norm);
        }
        std::sort(abs_flow.begin(), abs_flow.end());
        auto median = abs_flow[abs_flow.size() / 2];
        std::for_each(abs_flow.begin(), abs_flow.end(), [median](double &x) { x = std::abs(x - median); });
        std::sort(abs_flow.begin(), abs_flow.end());
        residual_threshold = abs_flow[abs_flow.size() / 2];
    }

    Eigen::VectorXd x_est(J.cols());
    std::vector<size_t> inlier_idxs;
    inlier_idxs.reserve(n_points);
    for (size_t iter{0}; iter <= n_iterations; ++iter) {
        // randomly select n_samples from data
        std::vector<size_t> sample_idxs(n_samples);
        for (size_t i{0}; i < n_samples; ++i)
            sample_idxs[i] = distr(gen);

        // solve
        solve_sampled(sample_idxs, x_est);

        Eigen::VectorXd error = J * x_est - flow_vectors;

        // compute inliers
        double error_sum{0};
        for (size_t i{0}; i < n_points; ++i) {
            auto error_x = error(i * 2);
            auto error_y = error(i * 2 + 1);
            double error_norm = std::sqrt(error_x * error_x + error_y * error_y);
            if (std::isnan(error_norm) || std::isinf(error_norm))
                continue;
            error_sum += error_norm;
            if (error_norm < residual_threshold)
                inlier_idxs.push_back(i);
        }
        error_sum /= (double) n_points;

//        if (error_sum < min_error) {
//            best_inliers = inlier_idxs;
//            min_error = error_sum;
//        }

        if (best_inliers.size() < inlier_idxs.size()) {
            best_inliers = inlier_idxs;
            min_error = error_sum;
        }

        if (static_cast<double>(best_inliers.size()) > 0.75 * static_cast<double>(n_points))
            break;

        inlier_idxs.clear();
    }
    std::cout << "Min error: " << min_error << std::endl;
    std::cout << "Best inliers size: " << best_inliers.size() << std::endl;
    assert(min_error < 200);

//    if (static_cast<double>(best_inliers.size()) < 0.5 * static_cast<double>(n_points)) {
//        cam_vel_est = Eigen::VectorXd::Zero(J.cols());
//        return;
//    }

    J_samples.resize(best_inliers.size() * 2, J.cols());
    flow_samples.resize(best_inliers.size() * 2);
    solve_sampled(best_inliers, cam_vel_est);
}

Eigen::Vector3d Estimator::update_flow_velocity(cv::Mat &frame, double time, const Eigen::Matrix3d &cam_R_enu,
                                                const Eigen::Vector3d &r, const Eigen::Matrix3d &K,
                                                const double height, const Eigen::Vector3d &omega) {
    if (!prev_frame_) {
        this->pre_frame_time_ = time;
        this->prev_frame_ = std::make_shared<cv::Mat>(frame);
        return {0, 0, 0};
    }

    cv::Mat flow;
    optflow_->calc(*prev_frame_, frame, flow);

//######################    DRAWING
    cv::Mat drawing_frame = frame.clone();
    for (int y = 0; y < drawing_frame.rows; y += 16) {
        for (int x = 0; x < drawing_frame.cols; x += 16) {
            // Get the flow from `flow`, which is a 2-channel matrix
            const cv::Point2f &fxy = flow.at<cv::Point2f>(y, x);
            // Draw lines on `drawing_frame` to represent flow
            cv::line(drawing_frame, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
                     cv::Scalar(0, 255, 0));
            cv::circle(drawing_frame, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
        }
    }
//######################    DRAWING

    int every_nth = 8;
    std::vector<cv::Point2f> flow_vecs;
    flow_vecs.reserve(frame.rows * frame.cols / (every_nth * every_nth));
    std::vector<cv::Point> samples;
    samples.reserve(frame.rows * frame.cols / (every_nth * every_nth));
    for (int row = every_nth; row < (frame.rows - every_nth); row += every_nth) {
        for (int col = every_nth; col < (frame.cols - every_nth); col += every_nth) {
            // Get the flow from `flow`, which is a 2-channel matrix
            const cv::Point2f &fxy = -flow.at<cv::Point2f>(row, col);
            flow_vecs.push_back(fxy);
            samples.emplace_back(row, col);
        }
    }

//######################    DRAWING
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

    Eigen::VectorXd depth(samples.size());
    Eigen::MatrixXd uv = Eigen::MatrixXd(2, samples.size());
    Eigen::VectorXd flow_eigen(2 * flow_vecs.size());
    const double dt = time - pre_frame_time_;
    for (size_t i = 0; i < samples.size(); i++) {
        // fill in pixel depths
        const Eigen::Vector3d Pt = compute_pixel_rel_position(
                Eigen::Vector2d(samples[i].x, samples[i].y), cam_R_enu, K, height, false);
        const double pixel_depth = Pt.norm();
        if (pixel_depth > 100)
            continue;
        if (flow_vecs[i].x == 0 && flow_vecs[i].y == 0)
            continue;
        depth(i) = pixel_depth;

        // store the uv coordinates
        uv(0, i) = static_cast<double>(samples[i].x);
        uv(1, i) = static_cast<double>(samples[i].y);

        flow_eigen(2 * i) = flow_vecs[i].x / dt;
        flow_eigen(2 * i + 1) = flow_vecs[i].y / dt;
    }
    Eigen::MatrixXd J; // Jacobian
    visjac_p(uv, depth, K, J);

    Eigen::VectorXd cam_vel_est;
    cam_vel_est.setZero();
    RANSAC_vel_regression(J, flow_eigen, cam_vel_est);

//    cam_vel_est = (J.transpose() * J).ldlt().solve(J.transpose() * flow_eigen);

    Eigen::Vector3d v_com_enu = cam_R_enu * cam_vel_est.segment(0, 3);
    Eigen::Vector3d w_com_enu = cam_R_enu * cam_vel_est.segment(3, 3);
    v_com_enu = v_com_enu - w_com_enu.cross(r);

//    v_com_enu -= omega.cross(r);

    // check for all zeros
    if (cam_vel_est.norm() > 1e-2 && kf_->is_initialized()) {
        // update C matrix
        static Eigen::MatrixXd C_vel(3, 12);
        C_vel << 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0;
        static Eigen::MatrixXd R_vel(3, 3);
        R_vel << 3.3457488e+00, 2.0952617e+00, 5.4171535e-01,
                2.0952617e+00, 5.3969640e+00, 7.7712646e-01,
                5.4171535e-01, 7.7712646e-01, 1.7101273e+00;

        kf_->update(v_com_enu, C_vel, R_vel);
    }

    this->pre_frame_time_ = time;
    *prev_frame_ = frame;
    return v_com_enu;
}

#endif