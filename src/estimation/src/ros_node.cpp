#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/range.hpp"
#include <vision_msgs/msg/detection2_d.hpp>
#include <Eigen/Dense>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include "px4_msgs/msg/vehicle_air_data.hpp"
#include "estimator.hpp"

class StateEstimationNode : public rclcpp::Node
{
public:
    StateEstimationNode() : Node("state_estimation_node")
    {
        bbox_sub_ = create_subscription<vision_msgs::msg::Detection2D>(
            "/bounding_box", 1, std::bind(&StateEstimationNode::bbox_callback, this, std::placeholders::_1));
        bbox_.bbox.size_x = -1;
        bbox_.bbox.size_y = -1;

        cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info", 1,
            std::bind(&StateEstimationNode::cam_info_callback, this, std::placeholders::_1));

        cam_target_pos_timer_ = create_wall_timer(
            std::chrono::milliseconds(33), std::bind(&StateEstimationNode::timer_callback, this));
        cam_target_pos_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
            "/cam_target_pos", 1);
        gt_pose_array_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
            "/gz/gt_pose_array", 1,
            std::bind(&StateEstimationNode::gt_pose_array_callback, this, std::placeholders::_1));
        gt_target_pos_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
            "/gt_target_pos", 1);

        range_sub_ = create_subscription<sensor_msgs::msg::Range>(
            "/teraranger_evo_40m", 1, std::bind(&StateEstimationNode::range_callback, this, std::placeholders::_1));

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
        qos.best_effort();
        air_data_sub_ = create_subscription<px4_msgs::msg::VehicleAirData>(
            "/fmu/out/vehicle_air_data", qos,
            std::bind(&StateEstimationNode::air_data_callback, this, std::placeholders::_1));

        tf_buffer_ =
            std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ =
            std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_timer_ = create_wall_timer(std::chrono::milliseconds(20), std::bind(&StateEstimationNode::tf_callback, this));

        estimator_ = std::make_unique<Estimator>();
    }

private:
    bool is_K_received()
    {
        return K_(0, 0) != 0;
    }

    void air_data_callback(const px4_msgs::msg::VehicleAirData::SharedPtr msg)
    {
        height_ = msg->baro_alt_meter;
    }

    void bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg)
    {
        bbox_ = *msg;
    }

    void cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        // only do once
        if (is_K_received())
            return;
        // TODO; simulation specific : (
        if (msg->header.frame_id == "x500_0/OakD-Lite/base_link/StereoOV7251")
            return;

        static constexpr double scale = .5;
        for (size_t i = 0; i < 9; i++)
        {
            size_t row = std::floor(i / 3);
            size_t col = i % 3;
            if (row == 0 || row == 1)
                K_(row, col) = msg->k[i] * scale;
            else
                K_(row, col) = msg->k[i];
        }
    }

    void gt_pose_array_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
        // take norm between points 0 and 1
        if (msg->poses.size() < 2)
            return;
        auto p0 = msg->poses[0];
        auto p1 = msg->poses[1];
        Eigen::Vector3d p0_vec(p0.position.x, p0.position.y, p0.position.z);
        Eigen::Vector3d p1_vec(p1.position.x, p1.position.y, p1.position.z);
        auto norm = (p0_vec - p1_vec).norm();
        auto gt_point = std::make_unique<geometry_msgs::msg::PointStamped>();
        gt_point->header.stamp = msg->header.stamp;
        gt_point->header.frame_id = "odom";
        gt_point->point.x = norm;
        gt_target_pos_pub_->publish(*gt_point);
    }

    void timer_callback()
    {
        // check that info arrived
        if (height_ < 0 || std::isnan(height_) || std::isinf(height_))
            return;
        if (bbox_.bbox.size_x < 0 || bbox_.bbox.size_y < 0)
            return;
        if (!is_K_received())
            return;
        if (!image_tf_ || !base_link_enu_) //  || !base_link_enu_ || !tera_tf_
            return;

        auto img_T_base = tf_msg_to_affine(*image_tf_);
        img_T_base.translation() = Eigen::Vector3d::Zero();
        auto base_T_odom = tf_msg_to_affine(*base_link_enu_);
        // auto tera_T_base = tf_msg_to_affine(*tera_tf_);

        Eigen::Vector2d uv_point;
        uv_point << bbox_.bbox.center.position.x,
            bbox_.bbox.center.position.y;

        // TODO:
        // Eigen::Vector3d H_vec;
        // H_vec << 0, 0, height_;
        // H_vec = base_T_odom * tera_T_base * H_vec;

        auto cam_R_enu = base_T_odom.rotation() * img_T_base.rotation();
        Eigen::Vector3d Pt = estimator_->compute_pixel_rel_position(uv_point, cam_R_enu, K_, height_);

        RCLCPP_INFO(this->get_logger(), "xyz: %f %f %f; norm: %f",
                    Pt[0], Pt[1], Pt[2], Pt.norm());

        // publish norm
        geometry_msgs::msg::PointStamped msg;
        msg.header.stamp = bbox_.header.stamp;
        msg.header.frame_id = "odom";
        msg.point.x = Pt.norm();
        cam_target_pos_pub_->publish(msg);
    }

    void range_callback(const sensor_msgs::msg::Range::SharedPtr msg)
    {
        height_ = msg->range;
    }

    void tf_lookup_helper(geometry_msgs::msg::TransformStamped &tf,
                          const std::string &target_frame, const std::string &source_frame)
    {
        try
        {
            tf = tf_buffer_->lookupTransform(
                target_frame, source_frame,
                tf2::TimePointZero);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_INFO(
                this->get_logger(), "Could not transform %s to %s: %s",
                source_frame.c_str(), target_frame.c_str(), ex.what());
        }
    }

    void tf_callback()
    {
        geometry_msgs::msg::TransformStamped image_tf;
        tf_lookup_helper(image_tf, "base_link", "camera_link_optical");
        image_tf_ = std::make_unique<geometry_msgs::msg::TransformStamped>(image_tf);

        // image_tf.transform.translation.x = -0.059;
        // image_tf.transform.translation.y = 0.031;
        // image_tf.transform.translation.z = -0.131;
        // image_tf.transform.rotation.x = 0.654;
        // image_tf.transform.rotation.y = -0.652;
        // image_tf.transform.rotation.z = 0.271;
        // image_tf.transform.rotation.w = 0.272;

        // tf_lookup_helper(tera_tf, "base_link", "teraranger_evo_40m");
        // geometry_msgs::msg::TransformStamped tera_tf;
        // tera_tf.transform.translation.x = -0.133;
        // tera_tf.transform.translation.y = 0.029;
        // tera_tf.transform.translation.z = -0.07;
        // tera_tf.transform.rotation.x = 0.0;
        // tera_tf.transform.rotation.y = 0.0;
        // tera_tf.transform.rotation.z = 0.0;
        // tera_tf.transform.rotation.w = 1.0;
        // tera_tf_ = std::make_unique<geometry_msgs::msg::TransformStamped>(tera_tf);

        geometry_msgs::msg::TransformStamped base_link_enu;
        tf_lookup_helper(base_link_enu, "odom", "base_link");
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

    rclcpp::TimerBase::SharedPtr cam_target_pos_timer_;
    rclcpp::TimerBase::SharedPtr tf_timer_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAirData>::SharedPtr air_data_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr cam_target_pos_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr gt_target_pos_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr gt_pose_array_sub_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<geometry_msgs::msg::TransformStamped> image_tf_{nullptr};
    std::unique_ptr<geometry_msgs::msg::TransformStamped> tera_tf_{nullptr};
    std::unique_ptr<geometry_msgs::msg::TransformStamped> base_link_enu_{nullptr};

    vision_msgs::msg::Detection2D bbox_;
    double height_{-1};
    Eigen::Matrix<double, 3, 3> K_;
    std::unique_ptr<Estimator> estimator_{nullptr};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StateEstimationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
