#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/range.hpp"
#include <vision_msgs/msg/detection2_d.hpp>
#include <Eigen/Dense>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "px4_msgs/msg/vehicle_air_data.hpp"

class StateEstimationNode : public rclcpp::Node
{
public:
    StateEstimationNode() : Node("state_estimation_node")
    {
        bbox_sub_ = create_subscription<vision_msgs::msg::Detection2D>(
            "/bounding_box", 10, std::bind(&StateEstimationNode::bbox_callback, this, std::placeholders::_1));
        bbox_.bbox.size_x = -1;
        bbox_.bbox.size_y = -1;

        cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info", 10, std::bind(&StateEstimationNode::cam_info_callback, this, std::placeholders::_1));

        cam_target_pos_timer_ = create_wall_timer(std::chrono::milliseconds(33), std::bind(&StateEstimationNode::timer_callback, this));

        range_sub_ = create_subscription<sensor_msgs::msg::Range>(
            "/teraranger_evo_40m", 10, std::bind(&StateEstimationNode::range_callback, this, std::placeholders::_1));

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
        qos.best_effort();
        air_data_sub_ = create_subscription<px4_msgs::msg::VehicleAirData>(
            "/fmu/out/vehicle_air_data", qos, std::bind(&StateEstimationNode::air_data_callback, this, std::placeholders::_1));

        tf_buffer_ =
            std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ =
            std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_timer_ = create_wall_timer(std::chrono::milliseconds(20), std::bind(&StateEstimationNode::tf_callback, this));
    }

private:
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
        if (msg->header.frame_id == "x500_0/OakD-Lite/base_link/StereoOV7251")
        {
            return;
        }
        cam_info_ = *msg;
    }

    void timer_callback()
    {
        // check that info arrived
        if (height_ < 0 || std::isnan(height_) || std::isinf(height_))
            return;
        if (bbox_.bbox.size_x < 0 || bbox_.bbox.size_y < 0)
            return;
        if (cam_info_.k.size() < 9)
            return;
        if (!image_tf_ || !base_link_enu_) //  || !base_link_enu_ || !tera_tf_
            return;

        auto img_T_base = tf_msg_to_affine(*image_tf_);
        img_T_base.translation() = Eigen::Vector3d::Zero();
        auto base_T_odom = tf_msg_to_affine(*base_link_enu_);
        // auto tera_T_base = tf_msg_to_affine(*tera_tf_);

        Eigen::Matrix<double, 3, 3> K;
        static constexpr double scale = .5;
        for (size_t i = 0; i < 9; i++)
        {
            size_t row = std::floor(i / 3);
            size_t col = i % 3;
            if (row == 0)
                K(row, col) = cam_info_.k[i] * scale;
            else if (row == 1)
                K(row, col) = cam_info_.k[i] * scale;
            else
                K(row, col) = cam_info_.k[i];
        }
        Eigen::Matrix<double, 3, 3> Kinv = K.inverse();
        std::cout << Kinv << std::endl;

        std::array<Eigen::Vector3d, 3> xyz_points;
        std::array<Eigen::Vector2d, 3> uv_points;
        uv_points[0] << bbox_.bbox.center.position.x,
            bbox_.bbox.center.position.y;
        uv_points[1] << bbox_.bbox.center.position.x + bbox_.bbox.size_x / 2,
            bbox_.bbox.center.position.y + bbox_.bbox.size_y / 2;
        uv_points[2] << bbox_.bbox.center.position.x - bbox_.bbox.size_x / 2,
            bbox_.bbox.center.position.y - bbox_.bbox.size_y / 2;
        Eigen::Vector3d H_vec;
        H_vec << 0, 0, height_;
        // H_vec = base_T_odom * tera_T_base * H_vec;
        Eigen::Vector3d lr;
        lr << 0, 0, -1;
        std::cout << img_T_base.rotation() << std::endl;
        for (size_t i = 0; auto &uv : uv_points)
        {
            Eigen::Vector3d Puv_hom;
            Puv_hom << uv[0], uv[1], 1;

            Eigen::Vector3d Pc = Kinv * Puv_hom; // R_f * 
            std::cout << Pc << std::endl;
            // base_T_odom *
            Eigen::Vector3d ls = img_T_base.rotation().inverse() * (Pc / Pc.norm());

            double d = H_vec[2] / (lr.transpose() * ls);

            // ls = base_T_odom * img_T_base.rotation() * (d * Pc / Pc.norm());

            Eigen::Vector3d Pt = ls * d;
            xyz_points[i] = Pt;

            RCLCPP_INFO(this->get_logger(), "xyz: %f %f %f; norm: %f",
                        xyz_points[i][0], xyz_points[i][1], xyz_points[i][2], xyz_points[i].norm());
            ++i;
        }
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
        geometry_msgs::msg::TransformStamped image_tf;
        tf_lookup_helper(image_tf, "camera_link_optical", "base_link");
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

    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAirData>::SharedPtr air_data_sub_;
    sensor_msgs::msg::CameraInfo cam_info_;
    vision_msgs::msg::Detection2D bbox_;
    rclcpp::TimerBase::SharedPtr cam_target_pos_timer_;
    rclcpp::TimerBase::SharedPtr tf_timer_;
    double height_{-1};

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    std::unique_ptr<geometry_msgs::msg::TransformStamped> image_tf_{nullptr};
    std::unique_ptr<geometry_msgs::msg::TransformStamped> tera_tf_{nullptr};
    std::unique_ptr<geometry_msgs::msg::TransformStamped> base_link_enu_{nullptr};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StateEstimationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
