#include "rclcpp/rclcpp.hpp"
#include "px4_msgs/msg/sensor_combined.hpp"
#include "px4_msgs/msg/vehicle_magnetometer.hpp"
#include "px4_msgs/msg/sensor_mag.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"

using std::placeholders::_1;

class SensorTranslator : public rclcpp::Node
{
public:
    SensorTranslator() : Node("sensor_translator")
    {
        imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", 10);
        mag_publisher_ = this->create_publisher<sensor_msgs::msg::MagneticField>("imu/mag", 10);
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
        qos.best_effort();
        sensor_combined_subscription_ = this->create_subscription<px4_msgs::msg::SensorCombined>(
            "/fmu/out/sensor_combined", qos, std::bind(&SensorTranslator::sensor_combined_callback, this, _1));
        vehicle_magnetometer_subscription_ = this->create_subscription<px4_msgs::msg::VehicleMagnetometer>(
            "/fmu/out/vehicle_magnetometer", qos, std::bind(&SensorTranslator::vehicle_magnetometer_callback, this, _1));
        sensor_mag_subscription_ = this->create_subscription<px4_msgs::msg::SensorMag>(
            "/fmu/out/sensor_mag", qos, std::bind(&SensorTranslator::sensor_mag_callback, this, _1));
    }

private:
    void sensor_combined_callback(const px4_msgs::msg::SensorCombined::SharedPtr msg)
    {
        auto imu_msg = std::make_unique<sensor_msgs::msg::Imu>();
        auto timestamp_microseconds = msg->timestamp;
        imu_msg->header.stamp = rclcpp::Time(timestamp_microseconds * 1000);
        imu_msg->header.frame_id = "base_link";
        imu_msg->linear_acceleration.x = static_cast<float>(msg->accelerometer_m_s2[0]);
        imu_msg->linear_acceleration.y = static_cast<float>(msg->accelerometer_m_s2[1]);
        imu_msg->linear_acceleration.z = static_cast<float>(msg->accelerometer_m_s2[2]);
        imu_msg->angular_velocity.x = static_cast<float>(msg->gyro_rad[0]);
        imu_msg->angular_velocity.y = static_cast<float>(msg->gyro_rad[1]);
        imu_msg->angular_velocity.z = static_cast<float>(msg->gyro_rad[2]);
        imu_publisher_->publish(std::move(imu_msg));
    }

    void vehicle_magnetometer_callback(const px4_msgs::msg::VehicleMagnetometer::SharedPtr msg)
    {
        auto mag_msg = std::make_unique<sensor_msgs::msg::MagneticField>();
        auto timestamp_microseconds = msg->timestamp;
        mag_msg->header.stamp = rclcpp::Time(timestamp_microseconds * 1000);
        mag_msg->header.frame_id = "base_link";
        mag_msg->magnetic_field.x = static_cast<float>(msg->magnetometer_ga[0]);
        mag_msg->magnetic_field.y = static_cast<float>(msg->magnetometer_ga[1]);
        mag_msg->magnetic_field.z = static_cast<float>(msg->magnetometer_ga[2]);
        mag_publisher_->publish(std::move(mag_msg));
    }

    void sensor_mag_callback(const px4_msgs::msg::SensorMag::SharedPtr msg)
    {
        auto mag_msg = std::make_unique<sensor_msgs::msg::MagneticField>();
        auto timestamp_microseconds = msg->timestamp;
        mag_msg->header.stamp = rclcpp::Time(timestamp_microseconds * 1000);
        mag_msg->header.frame_id = "base_link";
        mag_msg->magnetic_field.x = static_cast<float>(msg->x);
        mag_msg->magnetic_field.y = static_cast<float>(msg->y);
        mag_msg->magnetic_field.z = static_cast<float>(msg->z);
        mag_publisher_->publish(std::move(mag_msg));
    }

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr mag_publisher_;
    rclcpp::Subscription<px4_msgs::msg::SensorCombined>::SharedPtr sensor_combined_subscription_;
    rclcpp::Subscription<px4_msgs::msg::VehicleMagnetometer>::SharedPtr vehicle_magnetometer_subscription_;
    rclcpp::Subscription<px4_msgs::msg::SensorMag>::SharedPtr sensor_mag_subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto sensor_translator = std::make_shared<SensorTranslator>();
    rclcpp::spin(sensor_translator);
    rclcpp::shutdown();
    return 0;
}