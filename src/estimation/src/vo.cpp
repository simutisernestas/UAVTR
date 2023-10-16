#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <Eigen/Dense>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/opencv.hpp>
#include "cv_bridge/cv_bridge.h"
#include "filter.cpp"

class StateEstimationNode : public rclcpp::Node
{
public:
    StateEstimationNode() : Node("state_estimation_node")
    {
        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 10, std::bind(&StateEstimationNode::image_callback, this, std::placeholders::_1));
        cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info", 10, std::bind(&StateEstimationNode::cam_info_callback, this, std::placeholders::_1));
        range_sub_ = create_subscription<sensor_msgs::msg::Range>(
            "/teraranger_evo_40m", 10, std::bind(&StateEstimationNode::range_callback, this, std::placeholders::_1));
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data", 10, std::bind(&StateEstimationNode::imu_callback, this, std::placeholders::_1));
        // "/camera/imu", 10, std::bind(&StateEstimationNode::imu_callback, this, std::placeholders::_1));

        tf_buffer_ =
            std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ =
            std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_timer_ = create_wall_timer(std::chrono::milliseconds(20), std::bind(&StateEstimationNode::tf_callback, this));

        state.setZero();
        state[robot::N - 1] = 10.0;
        ekf.init(state);
    }

private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        if (!prev_imu_)
        {
            prev_imu_ = std::make_shared<sensor_msgs::msg::Imu>(*msg);
            return;
        }
        if (!base_link_enu_)
            return;

        // const rclcpp::Duration dt = rclcpp::Time(msg->header.stamp) - rclcpp::Time(prev_imu_->header.stamp);
        // const double dt_sec = dt.seconds();
        // RCLCPP_INFO(get_logger(), "Dt: %f", dt_sec);

        // should rotate to global frame plus subtract the gravity
        auto base_T_enu = tf_msg_to_affine(*base_link_enu_);
        Eigen::Vector3d acc_W;
        acc_W << msg->linear_acceleration.x,
            msg->linear_acceleration.y,
            msg->linear_acceleration.z;
        acc_W = base_T_enu * acc_W;
        acc_W[2] += 9.81;
        // RCLCPP_INFO(this->get_logger(), "acc_W: %f, %f, %f", acc_W[0], acc_W[1], acc_W[2]);

        robot::AccelerationMeasurement<double> measurement{};
        measurement << acc_W[0], acc_W[1], acc_W[2];
        state = ekf.predict(sys, control);
        state = ekf.update(acc_meas_model, measurement);

        prev_imu_ = std::make_shared<sensor_msgs::msg::Imu>(*msg);
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        if (!prev_image_)
        {
            prev_image_ = std::make_shared<cv::Mat>(cv_ptr->image);
            cv::cvtColor(*prev_image_, *prev_image_, cv::COLOR_RGB2GRAY);
            return;
        }
        if (!cam_info_.k.size())
            return;
        if (!base_link_enu_ || !image_tf_)
            return;

        // start
        auto start = std::chrono::high_resolution_clock::now();

        // Define camera intrinsic matrix
        cv::Mat K = cv::Mat_<double>(3, 3);
        for (int i = 0; i < 9; i++)
            K.at<double>(i) = cam_info_.k[i];

        cv::cvtColor(cv_ptr->image, cv_ptr->image, cv::COLOR_RGB2GRAY);

        if (!cv_ptr->image.isContinuous())
            cv_ptr->image = cv_ptr->image.clone();
        if (!prev_image_->isContinuous())
            *prev_image_ = prev_image_->clone();

        // detect features
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        detector->detect(cv_ptr->image, keypoints1);
        detector->detect(*prev_image_, keypoints2);

        // compute descriptors
        cv::Mat descriptors1, descriptors2;
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
        extractor->compute(cv_ptr->image, keypoints1, descriptors1);
        extractor->compute(*prev_image_, keypoints2, descriptors2);

        // match descriptors
        std::vector<cv::DMatch> matches;
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(descriptors1, descriptors2, matches);

        std::vector<cv::Point2f> good_points1, good_points2;
        for (size_t i = 0; i < matches.size(); i++)
        {
            good_points1.push_back(keypoints1[matches[i].queryIdx].pt);
            good_points2.push_back(keypoints2[matches[i].trainIdx].pt);
        }

        if (good_points1.size() < 8 || good_points2.size() < 8)
        {
            prev_image_ = std::make_shared<cv::Mat>(cv_ptr->image);
            return;
        }

        // RCLCPP_INFO(this->get_logger(), "good points: %lu", good_points1.size());
        // RCLCPP_INFO(this->get_logger(), "good points: %lu", good_points2.size());

        // compute essential matrix
        cv::Mat E = cv::findEssentialMat(good_points1, good_points2, K, cv::RANSAC, 0.999, 1.0);

        // recover pose
        cv::Mat R, t;
        cv::recoverPose(E, good_points1, good_points2, K, R, t);

        // TODO: must be in world
        // cam_T_base * -t; in the world frame
        auto cam_T_base = tf_msg_to_affine(*image_tf_);
        auto base_T_enu = tf_msg_to_affine(*base_link_enu_);
        Eigen::Vector3d t_vec;
        t_vec << -t.at<double>(0), -t.at<double>(1), -t.at<double>(2);
        Eigen::Vector3d t_world = base_T_enu * cam_T_base * t_vec;

        position_ += t_world;

        robot::PositionMeasurement<double> measurement{};
        measurement << position_[0], position_[1], position_[2];
        state = ekf.update(pos_meas_model, measurement);

        std::cout << "state: " << '\n'
                  << state << '\n';

        auto lambda = state[robot::N - 1];
        RCLCPP_INFO(this->get_logger(), "lambda: %f", lambda);

        prev_image_ = std::make_shared<cv::Mat>(cv_ptr->image);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        RCLCPP_INFO(this->get_logger(), "Duration: %ld", duration.count());
    }

    void cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        cam_info_ = *msg;
    }

    void range_callback(const sensor_msgs::msg::Range::SharedPtr msg)
    {
        // TODO: orientation in ENU
        height_ = msg->range;
    }

    void tf_lookup_helper(geometry_msgs::msg::TransformStamped &tf,
                          const std::string &from_frame, const std::string &to_frame)
    {
        try
        {
            tf = tf_buffer_->lookupTransform(
                from_frame, to_frame,
                tf2::TimePointZero);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_INFO(
                this->get_logger(), "Could not transform %s to %s: %s",
                to_frame.c_str(), from_frame.c_str(), ex.what());
        }
    }

    void tf_callback()
    {
        // tf_lookup_helper(image_tf, "base_link", "camera_color_optical_frame");
        geometry_msgs::msg::TransformStamped image_tf;
        image_tf.transform.translation.x = -0.059;
        image_tf.transform.translation.y = 0.031;
        image_tf.transform.translation.z = -0.131;
        image_tf.transform.rotation.x = 0.654;
        image_tf.transform.rotation.y = -0.652;
        image_tf.transform.rotation.z = 0.271;
        image_tf.transform.rotation.w = 0.272;
        image_tf_ = std::make_unique<geometry_msgs::msg::TransformStamped>(image_tf);

        // tf_lookup_helper(tera_tf, "base_link", "teraranger_evo_40m");
        geometry_msgs::msg::TransformStamped tera_tf;
        tera_tf.transform.translation.x = -0.133;
        tera_tf.transform.translation.y = 0.029;
        tera_tf.transform.translation.z = -0.07;
        tera_tf.transform.rotation.x = 0.0;
        tera_tf.transform.rotation.y = 0.0;
        tera_tf.transform.rotation.z = 0.0;
        tera_tf.transform.rotation.w = 1.0;
        tera_tf_ = std::make_unique<geometry_msgs::msg::TransformStamped>(tera_tf);

        geometry_msgs::msg::TransformStamped base_link_enu;
        tf_lookup_helper(base_link_enu, "base_link", "odom");
        base_link_enu_ = std::make_unique<geometry_msgs::msg::TransformStamped>(base_link_enu);
    }

    Eigen::Transform<double, 3, Eigen::Affine> tf_msg_to_affine(const geometry_msgs::msg::TransformStamped &tf_stamp)
    {
        Eigen::Quaterniond rotation(tf_stamp.transform.rotation.w,
                                    tf_stamp.transform.rotation.x,
                                    tf_stamp.transform.rotation.y,
                                    tf_stamp.transform.rotation.z);
        Eigen::Translation3d translation(tf_stamp.transform.translation.x,
                                         tf_stamp.transform.translation.y,
                                         tf_stamp.transform.translation.z);
        Eigen::Transform<double, 3, Eigen::Affine> transform = translation * rotation;
        return transform;
    }

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    sensor_msgs::msg::CameraInfo cam_info_;
    rclcpp::TimerBase::SharedPtr tf_timer_;
    double height_{-1};

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    std::unique_ptr<geometry_msgs::msg::TransformStamped> image_tf_{nullptr};
    std::unique_ptr<geometry_msgs::msg::TransformStamped> tera_tf_{nullptr};
    std::unique_ptr<geometry_msgs::msg::TransformStamped> base_link_enu_{nullptr};

    std::shared_ptr<cv::Mat> prev_image_{nullptr};
    std::shared_ptr<sensor_msgs::msg::Imu> prev_imu_{nullptr};

    Eigen::Vector3d position_{0, 0, 0};

    Kalman::ExtendedKalmanFilter<robot::State<double>> ekf{};
    robot::PositionMeasurementModel<double> pos_meas_model{};
    robot::AccelerationMeasurementModel<double> acc_meas_model{};
    robot::State<double> state{};
    robot::Control<double> control{};
    robot::SystemModel<double> sys{};
};

// int main()
// {
//     robot::State<double> state{};
//     state[robot::N - 1] = 1.0;
//     // print state
//     std::cout << "state: " << '\n'
//               << state << '\n';
//     robot::Control<double> control{};
//     robot::SystemModel<double> sys{};

//     Kalman::ExtendedKalmanFilter<robot::State<double>> ekf;
//     ekf.init(state);

//     ekf.predict(sys, control);

//     robot::PositionMeasurementModel<double> pos_meas_model{};

//     robot::PositionMeasurement<double> measurement{};

//     ekf.update(pos_meas_model, measurement);

//     sys.print_jacobians();

//     printf("Hello World!\n");
//     return 0;
// }

int main(int argc, char **argv)
{

    rclcpp::init(argc, argv);
    auto node = std::make_shared<StateEstimationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
