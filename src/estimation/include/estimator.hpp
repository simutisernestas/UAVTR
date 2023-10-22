#include "Eigen/Dense"
#include "opencv2/opencv.hpp"

class Estimator
{
public:
    Estimator();

    Eigen::Vector3d compute_pixel_rel_position(
        const Eigen::Vector2d &bbox_c, const Eigen::Matrix3d &cam_R_enu,
        const Eigen::Matrix3d &K, const double height);

    void update_flow_velocity(const cv::Mat &frame);

private:
    std::unique_ptr<cv::Mat> prev_frame_{nullptr};
    std::vector<cv::Point2f> p0_, p1_;
};