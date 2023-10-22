#include "estimator.hpp"

Estimator::Estimator()
{
    // Initialize the estimator
}

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
