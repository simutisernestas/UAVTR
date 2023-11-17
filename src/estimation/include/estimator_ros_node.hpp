#pragma once

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <vision_msgs/msg/detection2_d.hpp>
#include <Eigen/Dense>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include "px4_msgs/msg/vehicle_air_data.hpp"
#include "px4_msgs/msg/timesync_status.hpp"
#include "px4_msgs/msg/sensor_gps.hpp"
#include "estimator.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "visualization_msgs/msg/marker.hpp"

#define VEL_MEAS 1

class StateEstimationNode : public rclcpp::Node {
public:
    StateEstimationNode();

private:
    void gps_callback(px4_msgs::msg::SensorGps::SharedPtr msg);

    void timesync_callback(px4_msgs::msg::TimesyncStatus::SharedPtr msg);

    void imu_callback(sensor_msgs::msg::Imu::SharedPtr msg);

    inline bool is_K_received() { return K_(0, 0) != 0; }

    void air_data_callback(px4_msgs::msg::VehicleAirData::SharedPtr msg);

    void range_callback(sensor_msgs::msg::Range::SharedPtr msg);

    void bbox_callback(vision_msgs::msg::Detection2D::SharedPtr bbox);

    void cam_info_callback(sensor_msgs::msg::CameraInfo::SharedPtr msg);

    void gt_pose_array_callback(geometry_msgs::msg::PoseArray::SharedPtr msg);

    bool tf_lookup_helper(geometry_msgs::msg::TransformStamped &tf,
                          const std::string &target_frame, const std::string &source_frame,
                          const rclcpp::Time &time = rclcpp::Time(0));

    void tf_callback();

    static Eigen::Transform<double, 3, Eigen::Affine> tf_msg_to_affine(geometry_msgs::msg::TransformStamped &tf_stamp);

#if VEL_MEAS

    void img_callback(sensor_msgs::msg::Image::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

#endif

    rclcpp::TimerBase::SharedPtr tf_timer_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAirData>::SharedPtr air_data_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr cam_target_pos_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr gt_target_pos_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr imu_world_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr gt_pose_array_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_cam_sub_;
    rclcpp::Subscription<px4_msgs::msg::TimesyncStatus>::SharedPtr timesync_sub_;
    rclcpp::Subscription<px4_msgs::msg::SensorGps>::SharedPtr gps_sub_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr gps_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vec_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr state_pub_;

    rclcpp::CallbackGroup::SharedPtr vel_meas_callback_group_;
    rclcpp::CallbackGroup::SharedPtr target_bbox_callback_group_;
    rclcpp::CallbackGroup::SharedPtr imu_callback_group_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<geometry_msgs::msg::TransformStamped> image_tf_{nullptr};
    std::unique_ptr<geometry_msgs::msg::TransformStamped> tera_tf_{nullptr};

    double prev_imu_time_s = -1;
    std::atomic<double> offset_{0};
    std::atomic<double> height_{-1};
    Eigen::Matrix<double, 3, 3> K_;
    std::unique_ptr<Estimator> estimator_{nullptr};
    bool simulation_;
    std::mutex mtx_;
}; // class StateEstimationNode
