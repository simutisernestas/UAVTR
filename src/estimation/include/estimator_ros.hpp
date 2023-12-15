#pragma once

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <vision_msgs/msg/detection2_d.hpp>
#include <Eigen/Dense>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include "px4_msgs/msg/vehicle_air_data.hpp"
#include "px4_msgs/msg/timesync_status.hpp"
#include "estimator.hpp"
#include <sensor_msgs/msg/image.hpp>
#include "visualization_msgs/msg/marker.hpp"
#include "image_geometry/pinhole_camera_model.h"
#include "rclcpp/rclcpp.hpp"
#include "angvel_accum.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"

class StateEstimationNode : public rclcpp::Node {
public:
    StateEstimationNode();

private:
    void timesync_callback(px4_msgs::msg::TimesyncStatus::SharedPtr msg);

    void imu_callback(sensor_msgs::msg::Imu::SharedPtr msg);

    void cam_imu_callback(sensor_msgs::msg::Imu::SharedPtr msg);

    inline bool is_K_received() { return K_(0, 0) != 0; }

    void air_data_callback(px4_msgs::msg::VehicleAirData::SharedPtr msg);

    void range_callback(sensor_msgs::msg::Range::SharedPtr msg);

    void bbox_callback(vision_msgs::msg::Detection2D::SharedPtr bbox);

    void cam_info_callback(sensor_msgs::msg::CameraInfo::SharedPtr msg);

    bool tf_lookup_helper(geometry_msgs::msg::TransformStamped &tf,
                          const std::string &target_frame, const std::string &source_frame,
                          const rclcpp::Time &time = rclcpp::Time(0));

    void tf_callback();

    void state_pub_callback();

    rclcpp::Time get_correct_fusion_time(const std_msgs::msg::Header &header, bool use_offset);

    static Eigen::Transform<float, 3, Eigen::Affine> tf_msg_to_affine(geometry_msgs::msg::TransformStamped &tf_stamp);

    void img_callback(sensor_msgs::msg::Image::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

    rclcpp::TimerBase::SharedPtr tf_timer_;
    rclcpp::TimerBase::SharedPtr state_pub_timer_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAirData>::SharedPtr air_data_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<px4_msgs::msg::TimesyncStatus>::SharedPtr timesync_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr cam_imu_sub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr state_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr target_pt_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr range_pub_;

    rclcpp::CallbackGroup::SharedPtr vel_meas_callback_group_;
    rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
    rclcpp::CallbackGroup::SharedPtr rest_sensors_callback_group_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<geometry_msgs::msg::TransformStamped> image_tf_{nullptr};
    std::unique_ptr<geometry_msgs::msg::TransformStamped> tera_tf_{nullptr};

    std::atomic<double> offset_{0};
    Eigen::Matrix<float, 3, 3> K_;
    std::unique_ptr<Estimator> estimator_{nullptr};
    bool simulation_{false};
    std::atomic<bool> target_in_sight_{false};
    std::mutex mtx_;
    image_geometry::PinholeCameraModel cam_model_;
    std::unique_ptr<AngVelAccumulator> cam_ang_vel_accumulator_{nullptr};
    std::unique_ptr<AngVelAccumulator> drone_ang_vel_accumulator_{nullptr};
    float baro_ground_ref_;
    std::atomic<long> time_{0};
}; // class StateEstimationNode
