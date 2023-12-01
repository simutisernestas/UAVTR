#include "estimator.hpp"
#include <Eigen/Sparse>
#include <numeric>
#include <random>
#include <cassert>

#ifdef SAVEOUT
#include <fstream>
#endif

Estimator::Estimator() {
    const double dt = 1.0 / 128.0;
    Eigen::MatrixXd A(12, 12);
    get_A(A, dt);

    Eigen::MatrixXd Q(12, 12);
    Q.block(9, 9, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
    Eigen::Matrix3d acc_variance;
    acc_variance << 3.182539, 0, 0,
            0, 3.387015, 0,
            0, 0, 1.540428;
    // TODO: tune this properly by the book
    auto Q33 = Eigen::MatrixXd::Identity(3, 3) * std::pow(dt, 4) / 4.0;
    auto Q36 = Eigen::MatrixXd::Identity(3, 3) * std::pow(dt, 3) / 2.0;
    auto Q39 = Eigen::MatrixXd::Identity(3, 3) * std::pow(dt, 2) / 2.0;
    auto Q66 = Eigen::MatrixXd::Identity(3, 3) * std::pow(dt, 2);
    auto Q69 = Eigen::MatrixXd::Identity(3, 3) * dt;
    Q.block(0, 0, 3, 3) = Q33 * acc_variance;
    Q.block(0, 3, 3, 3) = Q36 * acc_variance;
    Q.block(3, 0, 3, 3) = Q36 * acc_variance;
    Q.block(0, 6, 3, 3) = Q39 * acc_variance;
    Q.block(6, 0, 3, 3) = Q39 * acc_variance;
    Q.block(3, 3, 3, 3) = Q66 * acc_variance;
    Q.block(3, 6, 3, 3) = Q69 * acc_variance;
    Q.block(6, 3, 3, 3) = Q69 * acc_variance;
    Q.block(6, 6, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * acc_variance;

    // relativee position measurement
    Eigen::MatrixXd C(2, 12);
    C << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    Eigen::MatrixXd R(2, 2);
    R << 1.7650208e+01, 1.2699096e+01,
            1.2699096e+01, 1.5283947e+01;

    Eigen::MatrixXd P(12, 12);
    P = Eigen::MatrixXd::Identity(12, 12) * 100.0;

    kf_ = std::make_unique<KalmanFilter>(A, C, Q, R, P);

    const std::array<double, 3> a = {1.0, -1.56101808, 0.64135154};
    const std::array<double, 3> b = {0.02008337, 0.04016673, 0.02008337};
    // Create a low pass filter objects.
    for (int i = 0; i < 3; i++)
        lp_acc_filter_arr_[i] = std::make_unique<LowPassFilter<double, 3 >>(b, a);

    optflow_->setGridStep({6, 6}); // increasing this reduces runtime
}

void Estimator::get_A(Eigen::MatrixXd &A, double dt) {
    A.setZero();
    // incorporate IMU after tests
    double ddt2 = dt * dt * .5;
    A << 1, 0, 0, dt, 0, 0, ddt2, 0, 0, -ddt2, 0, 0,
            0, 1, 0, 0, dt, 0, 0, ddt2, 0, 0, -ddt2, 0,
            0, 0, 1, 0, 0, dt, 0, 0, ddt2, 0, 0, -ddt2,
            0, 0, 0, 1, 0, 0, dt, 0, 0, -dt, 0, 0,
            0, 0, 0, 0, 1, 0, 0, dt, 0, 0, -dt, 0,
            0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, -dt,
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
    R_accel << 3.182539, 0, 0,
            0, 3.387015, 0,
            0, 0, 1.540428;
    kf_->update(accel, C_accel, R_accel);
}

void Estimator::visjac_p(const Eigen::MatrixXd &uv,
                         const Eigen::VectorXd &depth,
                         const Eigen::Matrix3d &K,
                         Eigen::MatrixXd &L) {
    assert(uv.cols() == depth.size());

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
    const size_t n_iterations = 4000;
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

    assert(J.cols() == 3);
    Eigen::VectorXd x_est(J.cols());
    Eigen::VectorXd best_x_est(J.cols());
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
        for (long i{0}; i < static_cast<long>(n_points); ++i) {
            const double error_x = error(i * 2);
            const double error_y = error(i * 2 + 1);
            const double error_norm = std::abs(error_x) + std::abs(error_y);
            if (std::isnan(error_norm) || std::isinf(error_norm)) {
                std::cout << "error norm is nan or inf" << std::endl;
                cam_vel_est = Eigen::VectorXd::Zero(J.cols());
                return;
            }
            if (error_norm < 3) {
                inlier_idxs.push_back(i);
                error_sum += error_norm;
            }
        }
        error_sum /= (double) inlier_idxs.size();

        if (best_inliers.size() < inlier_idxs.size()) {
            best_inliers = inlier_idxs;
            min_error = error_sum;
            std::cout << "iteration " << iter << std::endl;
            std::cout << "Min error: " << min_error << std::endl;
            std::cout << "Best inliers size: " << best_inliers.size() << std::endl;
        }

        if (static_cast<double>(best_inliers.size()) > 0.75 * static_cast<double>(n_points))
            break;

        inlier_idxs.clear();
    }

    J_samples.resize(best_inliers.size() * 2, J.cols());
    flow_samples.resize(best_inliers.size() * 2);
    solve_sampled(best_inliers, cam_vel_est);
}

Eigen::Vector3d Estimator::update_flow_velocity(cv::Mat &frame, double time, const Eigen::Matrix3d &cam_R_enu,
                                                const Eigen::Vector3d &r, const Eigen::Matrix3d &K,
                                                const double height, const Eigen::Vector3d &omega,
                                                const Eigen::Vector3d &drone_omega) {
    if (!prev_frame_) {
        this->pre_frame_time_ = time;
        this->prev_frame_ = std::make_shared<cv::Mat>(frame);
        return {0, 0, 0};
    }

    const double dt = time - pre_frame_time_;
    assert(dt > 0);
    if (dt > .75) {
        this->pre_frame_time_ = time;
        *prev_frame_ = frame;
        return {0, 0, 0};
    }
// TODO: need to improve the runtime of this significantly !!!
// TODO: move to float32 everwhere
// TODO: update opencv ?
// TODO: could investigate resizing the image
// TODO: what's the type of the image? does it work with lower precision maybe f16?
// TODO: profile it

#ifdef SAVEOUT
    static int count{0};
    count++;
    if (count == 1) {
        // dump everything to a file
        cv::imwrite("/tmp/frame0.png", *prev_frame_);
        cv::imwrite("/tmp/frame1.png", frame);
        // open a text file
        std::ofstream file("/tmp/flowinfo.txt");
        file << "time:" << time << std::endl;
        file << "prev_time:" << pre_frame_time_ << std::endl;
        file << "cam_R_enu:" << cam_R_enu << std::endl;
        file << "height:" << height << std::endl;
        file << "r:" << r << std::endl;
        file << "K:" << std::endl << K << std::endl;
        exit(0);
    }
#endif

    cv::Mat flow;
    optflow_->calc(*prev_frame_, frame, flow);

    int every_nth = 8;
    std::vector<cv::Point2f> flow_vecs;
    flow_vecs.reserve(frame.rows * frame.cols / (every_nth * every_nth));
    std::vector<cv::Point> samples;
    samples.reserve(frame.rows * frame.cols / (every_nth * every_nth));
    // the multiplies are orientation dependant
    for (int row = (int) every_nth; row < (frame.rows - every_nth); row += every_nth) {
        for (int col = (int) every_nth; col < (frame.cols - every_nth); col += every_nth) {
            // Get the flow from `flow`, which is a 2-channel matrix
            const cv::Point2f &fxy = flow.at<cv::Point2f>(row, col);
            flow_vecs.push_back(fxy);
            samples.emplace_back(row, col);
        }
    }

//######################    DRAWING
    cv::Mat drawing_frame = frame.clone();
    for (int y = 0; y < drawing_frame.rows; y += every_nth) {
        for (int x = 0; x < drawing_frame.cols; x += every_nth) {
            // Get the flow from `flow`, which is a 2-channel matrix
            const cv::Point2f &fxy = flow.at<cv::Point2f>(y, x);
            // Draw lines on `drawing_frame` to represent flow
            cv::line(drawing_frame, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
                     cv::Scalar(0, 255, 0));
            cv::circle(drawing_frame, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
        }
    }
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

    const double MAX_Z = 40;
    Eigen::VectorXd depth(samples.size());
    Eigen::MatrixXd uv = Eigen::MatrixXd(2, samples.size());
    Eigen::VectorXd flow_eigen(2 * flow_vecs.size());
    long insert_idx = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        bool is_flow_present = (flow_vecs[i].x != 0 && flow_vecs[i].y != 0);
        if (!is_flow_present)
            continue;

        const double Z = get_pixel_z_in_camera_frame(
                Eigen::Vector2d(samples[i].x, samples[i].y), cam_R_enu, K, height);
        if (Z < 0 || Z > MAX_Z)
            continue;

        depth(insert_idx) = Z;
        uv(0, insert_idx) = static_cast<double>(samples[i].x);
        uv(1, insert_idx) = static_cast<double>(samples[i].y);
        flow_eigen(2 * insert_idx) = flow_vecs[i].x / dt;
        flow_eigen(2 * insert_idx + 1) = flow_vecs[i].y / dt;
        ++insert_idx;
    }
    if (insert_idx == 0) {
        this->pre_frame_time_ = time;
        *prev_frame_ = frame;
        return {0, 0, 0};
    }

    // resize the matrices to fit the filled values
    depth.conservativeResize(insert_idx);
    uv.conservativeResize(Eigen::NoChange, insert_idx);
    flow_eigen.conservativeResize(2 * insert_idx);

    Eigen::MatrixXd J; // Jacobian
    visjac_p(uv, depth, K, J);

//    // subtract angular velocity part of the flow, to improve accuracy
//    Eigen::Vector3d z_vel = {0, 0, -0.24866668}; // enu
//    // convert to camera frame
//    z_vel = cam_R_enu.transpose() * z_vel;
    for (long i = 0; i < J.rows(); i++) {
        Eigen::Vector3d Jw = {J(i, 3), J(i, 4), J(i, 5)};
        flow_eigen(i) -= Jw.dot(omega);
//        Eigen::Vector3d Jv = {J(i, 0), J(i, 1), J(i, 2)};
//        flow_eigen(i) -= Jv.dot(z_vel);
//        auto v_due_to_angular_velocity = drone_omega.cross(r);
//        flow_eigen(i) -= Jv.dot(v_due_to_angular_velocity);
    }

    Eigen::VectorXd cam_vel_est;
    RANSAC_vel_regression(J.block(0, 0, J.rows(), 3), flow_eigen, cam_vel_est);

    Eigen::Vector3d v_com_enu = cam_R_enu * cam_vel_est.segment(0, 3);
    v_com_enu -= drone_omega.cross(r);

    if (cam_vel_est.norm() > 1e-1 && kf_->is_initialized()) {
        static Eigen::MatrixXd C_vel(2, 12);
        C_vel << 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0;
        static Eigen::MatrixXd R_vel(2, 2);
        R_vel << 1.0, 0,
                0, 1.0;

        kf_->update(v_com_enu.segment(0, 2), C_vel, R_vel);
    }

    this->pre_frame_time_ = time;
    *prev_frame_ = frame;
    return v_com_enu;
}

void Estimator::compute_velocity(const Eigen::MatrixXd &J,
                                 const Eigen::VectorXd &flow,
                                 Eigen::VectorXd &vel) {
    vel = (J.transpose() * J).ldlt().solve(J.transpose() * flow);
}
