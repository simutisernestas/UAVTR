#pragma once

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
#include "image_geometry/pinhole_camera_model.h"

#define RCLCPP__NODE_IMPL_HPP_

#include "rclcpp/node.hpp"

struct CamAngVelAccumulator {
    CamAngVelAccumulator() : x(0), y(0), z(0),
                             ang_vel_count(0) {}

    void add(const float x_add, const float y_add, const float z_add) {
        std::scoped_lock lock(mtx);
        x += x_add;
        y += y_add;
        z += z_add;
        ang_vel_count++;
    }

    void reset() {
        x = y = z = 0;
        ang_vel_count = 0;
    }

    [[nodiscard]] Eigen::Vector3f get_ang_vel() {
        std::scoped_lock lock(mtx);

        if (ang_vel_count == 0)
            return {0, 0, 0};

        Eigen::Vector3f ang_vel{x / (float) ang_vel_count,
                                y / (float) ang_vel_count,
                                z / (float) ang_vel_count};
        reset();

        return ang_vel;
    }

    float x, y, z;
    uint16_t ang_vel_count;
    std::mutex mtx;
};

class StateEstimationNode : public rclcpp::Node {
public:
    StateEstimationNode();

private:
    void gps_callback(px4_msgs::msg::SensorGps::SharedPtr msg);

    void timesync_callback(px4_msgs::msg::TimesyncStatus::SharedPtr msg);

    void imu_callback(sensor_msgs::msg::Imu::SharedPtr msg);

    void cam_imu_callback(sensor_msgs::msg::Imu::SharedPtr msg);

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

    rclcpp::Time get_correct_fusion_time(const std_msgs::msg::Header &header, bool use_offset);

    static Eigen::Transform<float, 3, Eigen::Affine> tf_msg_to_affine(geometry_msgs::msg::TransformStamped &tf_stamp);

    void img_callback(sensor_msgs::msg::Image::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

    rclcpp::TimerBase::SharedPtr tf_timer_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAirData>::SharedPtr air_data_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr gt_pose_array_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_cam_sub_;
    rclcpp::Subscription<px4_msgs::msg::TimesyncStatus>::SharedPtr timesync_sub_;
    rclcpp::Subscription<px4_msgs::msg::SensorGps>::SharedPtr gps_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr cam_imu_sub_;

    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr cam_target_pos_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr gt_target_pos_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr imu_world_pub_;
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
    Eigen::Matrix<float, 3, 3> K_;
    std::unique_ptr<Estimator> estimator_{nullptr};
    bool simulation_{false};
    std::mutex mtx_;
    image_geometry::PinholeCameraModel cam_model_;
    std::unique_ptr<CamAngVelAccumulator> cam_ang_vel_accumulator_{nullptr};
}; // class StateEstimationNode
