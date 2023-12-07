#include "px4_msgs/msg/sensor_combined.hpp"
#include "px4_msgs/msg/vehicle_attitude.hpp"
#include "px4_msgs/msg/sensor_mag.hpp"
#include "px4_msgs/msg/vehicle_magnetometer.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"
#include "Fusion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <Eigen/Dense>
#include "FusionMath.h"
#define RCLCPP__NODE_IMPL_HPP_
#include "rclcpp/node.hpp"

using std::placeholders::_1;

#define SAMPLE_RATE (128)

class SensorTranslator : public rclcpp::Node {
public:
    SensorTranslator() : Node("sensor_translator"), tf_broadcaster_(this) {
        imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", 10);
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
        qos.best_effort();
        sensor_combined_subscription_ = this->create_subscription<px4_msgs::msg::SensorCombined>(
                "/fmu/out/sensor_combined", qos, std::bind(&SensorTranslator::sensor_combined_callback, this, _1));
        sensor_mag_subscription_ = this->create_subscription<px4_msgs::msg::SensorMag>(
                "/fmu/out/sensor_mag", qos, std::bind(&SensorTranslator::sensor_mag_callback, this, _1));
        vehicle_mag_subscription_ = this->create_subscription<px4_msgs::msg::VehicleMagnetometer>(
                "/fmu/out/vehicle_magnetometer", qos, std::bind(&SensorTranslator::vehicle_mag_callback, this, _1));

        FusionOffsetInitialise(&offset, SAMPLE_RATE);
        FusionAhrsInitialise(&ahrs);
        FusionAhrsSetSettings(&ahrs, &settings);

#define DEBUG 1
#ifdef DEBUG
        euler_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/euler", 10);
        euler_publisher_px4_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/euler_px4", 10);
        vehicle_attitude_subscription_ = this->create_subscription<px4_msgs::msg::VehicleAttitude>(
            "/fmu/out/vehicle_attitude", qos, std::bind(&SensorTranslator::vehicle_attitude_callback, this, _1));
#endif
    }

private:
    // Fusion filter settings
    const FusionMatrix gyroscopeMisalignment = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    const FusionVector gyroscopeSensitivity = {1.0f, 1.0f, 1.0f};
    const FusionVector gyroscopeOffset = {0.0f, 0.0f, 0.0f};
    const FusionMatrix accelerometerMisalignment = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    const FusionVector accelerometerSensitivity = {1.0f, 1.0f, 1.0f};
    const FusionVector accelerometerOffset = {0.0f, 0.0f, 0.0f};
    const FusionMatrix softIronMatrix = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    const FusionVector hardIronOffset = {0.0f, 0.0f, 0.0f};
    const FusionAhrsSettings settings = {
            .convention = FusionConventionEnu,
            .gain = 0.8f, // low gain will trust gyro, susceptible to drift
            .gyroscopeRange = 2000.0f, /* replace this with actual gyroscope range in degrees/s */
            .accelerationRejection = 10.0f,
            .magneticRejection = 10.0f,
            .recoveryTriggerPeriod = 5 * SAMPLE_RATE, /* 5 seconds */
    };
    FusionOffset offset;
    FusionAhrs ahrs;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    ///////// funcs from mavros ftf_frame_conversions.cpp
    Eigen::Quaternionf quaternion_from_rpy(const Eigen::Vector3f &rpy) {
        // YPR - ZYX
        return Eigen::Quaternionf(
                Eigen::AngleAxisf(rpy.z(), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(rpy.y(), Eigen::Vector3f::UnitY()) *
                Eigen::AngleAxisf(rpy.x(), Eigen::Vector3f::UnitX()));
    }

    inline Eigen::Quaternionf quaternion_from_rpy(
            const float roll, const float pitch,
            const float yaw) {
        return quaternion_from_rpy(Eigen::Vector3f(roll, pitch, yaw));
    }
    ///////// funcs from mavros ftf_frame_conversions.cpp

    void sensor_combined_callback(const px4_msgs::msg::SensorCombined::SharedPtr msg) {
        if (last_timestamp_ == 0) {
            last_timestamp_ = msg->timestamp;
            return;
        }

        static const auto AIRCRAFT_BASELINK_Q = quaternion_from_rpy(M_PI, 0.0, 0.0);
        static const Eigen::Affine3f AIRCRAFT_BASELINK_AFFINE(AIRCRAFT_BASELINK_Q);

        // AIRCRAFT_BASELINK_AFFINE * vec; need to do this for gyro
        Eigen::Vector3f gyro(msg->gyro_rad[0], msg->gyro_rad[1], msg->gyro_rad[2]);
        gyro = AIRCRAFT_BASELINK_AFFINE * gyro;
        // same for acceleration
        Eigen::Vector3f accel(msg->accelerometer_m_s2[0], msg->accelerometer_m_s2[1], msg->accelerometer_m_s2[2]);
        accel = AIRCRAFT_BASELINK_AFFINE * accel;
        // mag
        Eigen::Vector3f mag(mag_msg_.magnetic_field.x, mag_msg_.magnetic_field.y, mag_msg_.magnetic_field.z);
        mag = AIRCRAFT_BASELINK_AFFINE * mag;

        // prepare data for fusion
        const static float rad2deg = 180.0f / M_PI;
        FusionVector gyroscope = {gyro[0] * rad2deg, 
                                  gyro[1] * rad2deg, 
                                  gyro[2] * rad2deg};
        const static float g = 9.81555f;
        accel /= g;
        FusionVector accelerometer = {accel[0], accel[1], accel[2]};
        FusionVector magnetometer = {mag[0], mag[1], mag[2]};

        // Calculate delta time
        const uint64_t delta = msg->timestamp - last_timestamp_; // microseconds
        const double dt = (float) delta / 1000000.0f;              // seconds
        assert(dt > 0.0f && dt < 1.0);
        last_timestamp_ = msg->timestamp;

        if (mag_msg_.header.stamp.sec > 0) {
            // Update gyroscope AHRS algorithm
            FusionAhrsUpdate(&ahrs, gyroscope, accelerometer, magnetometer, dt);
            mag_msg_.header.stamp.sec = -1;
        } else {
            FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer, dt);
        }

        // publish tf
        const FusionQuaternion quaternion = FusionAhrsGetQuaternion(&ahrs);
        // const FusionQuaternion ned_q_enu = {-quaternion.element.w, quaternion.element.y,
        //     quaternion.element.x, -quaternion.element.z};
        geometry_msgs::msg::TransformStamped transform{};
        transform.header.stamp = rclcpp::Time(msg->timestamp * 1000);
        transform.header.frame_id = "odom";
        transform.child_frame_id = "base_link";
        transform.transform.rotation.w = quaternion.element.w;
        transform.transform.rotation.x = quaternion.element.x;
        transform.transform.rotation.y = quaternion.element.y;
        transform.transform.rotation.z = quaternion.element.z;
        tf_broadcaster_.sendTransform(transform);

        // Get acceleration in world frame
        const FusionVector earth = FusionAhrsGetEarthAcceleration(&ahrs);


        // publish IMU
        sensor_msgs::msg::Imu imu_msg{};
        auto timestamp_microseconds = msg->timestamp;
        imu_msg.header.stamp = rclcpp::Time(timestamp_microseconds * 1000);
        imu_msg.header.frame_id = "odom";
        imu_msg.linear_acceleration.x = earth.axis.x * g;
        imu_msg.linear_acceleration.y = earth.axis.y * g;
        imu_msg.linear_acceleration.z = earth.axis.z * g;
        imu_msg.angular_velocity.x = gyro[0];
        imu_msg.angular_velocity.y = gyro[1];
        imu_msg.angular_velocity.z = gyro[2];
        imu_publisher_->publish(imu_msg);

#ifdef DEBUG
        const FusionEuler euler = FusionQuaternionToEuler(quaternion);
        // construct msg
        sensor_msgs::msg::Imu euler_msg{};
        euler_msg.header.stamp = rclcpp::Time(msg->timestamp * 1000);
        euler_msg.header.frame_id = "base_link";
        euler_msg.linear_acceleration.x = earth.axis.x;
        euler_msg.linear_acceleration.y = earth.axis.y;
        euler_msg.linear_acceleration.z = earth.axis.z;
        euler_msg.angular_velocity.x = euler.angle.roll;
        euler_msg.angular_velocity.y = euler.angle.pitch;
        euler_msg.angular_velocity.z = euler.angle.yaw + 90.0f;
        euler_publisher_->publish(euler_msg);
#endif
    }

    void sensor_mag_callback(const px4_msgs::msg::SensorMag::SharedPtr msg) {
        sensor_msgs::msg::MagneticField mag_msg{};
        auto timestamp_microseconds = msg->timestamp;
        mag_msg.header.stamp = rclcpp::Time(timestamp_microseconds * 1000);
        mag_msg.header.frame_id = "base_link";
        mag_msg.magnetic_field.x = msg->x;
        mag_msg.magnetic_field.y = msg->y;
        mag_msg.magnetic_field.z = msg->z;
        mag_msg_ = mag_msg;
    }

    void vehicle_mag_callback(const px4_msgs::msg::VehicleMagnetometer::SharedPtr msg) {
        sensor_msgs::msg::MagneticField mag_msg{};
        auto timestamp_microseconds = msg->timestamp;
        mag_msg.header.stamp = rclcpp::Time(timestamp_microseconds * 1000);
        mag_msg.header.frame_id = "base_link";
        mag_msg.magnetic_field.x = msg->magnetometer_ga[0];
        mag_msg.magnetic_field.y = msg->magnetometer_ga[1];
        mag_msg.magnetic_field.z = msg->magnetometer_ga[2];
        mag_msg_ = mag_msg;
    }

#ifdef DEBUG
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr euler_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr euler_publisher_px4_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr vehicle_attitude_subscription_;
    void vehicle_attitude_callback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg)
    {
        const float x = msg->q[0];
        const float y = msg->q[1];
        const float z = msg->q[2];
        const float w = msg->q[3];

        FusionQuaternion quaternion = {w, x, y, z};
        const FusionEuler euler = FusionQuaternionToEuler(quaternion);

        sensor_msgs::msg::Imu euler_msg{};
        euler_msg.header.stamp = rclcpp::Time(msg->timestamp * 1000);
        euler_msg.header.frame_id = "base_link";
        euler_msg.angular_velocity.x = euler.angle.roll;
        euler_msg.angular_velocity.y = euler.angle.pitch;
        euler_msg.angular_velocity.z = euler.angle.yaw;
        euler_publisher_px4_->publish(euler_msg);
    }
#endif

    uint64_t last_timestamp_{0};
    sensor_msgs::msg::MagneticField mag_msg_{};
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Subscription<px4_msgs::msg::SensorCombined>::SharedPtr sensor_combined_subscription_;
    rclcpp::Subscription<px4_msgs::msg::SensorMag>::SharedPtr sensor_mag_subscription_;
    rclcpp::Subscription<px4_msgs::msg::VehicleMagnetometer>::SharedPtr vehicle_mag_subscription_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto sensor_translator = std::make_shared<SensorTranslator>();
    rclcpp::executors::StaticSingleThreadedExecutor executor;
    executor.add_node(sensor_translator);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}