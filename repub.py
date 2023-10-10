import rclpy
from rclpy.node import Node
from px4_msgs.msg import SensorCombined, VehicleMagnetometer, SensorMag
from sensor_msgs.msg import Imu, MagneticField
from rclpy.time import Time

# Orientation estimates
# ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args -p world_frame:=ned


class SensorTranslator(Node):
    def __init__(self):
        super().__init__('sensor_translator')
        self.imu_publisher = self.create_publisher(Imu, 'imu/data_raw', 10)
        self.mag_publisher = self.create_publisher(
            MagneticField, 'imu/mag', 10)
        qos = rclpy.qos.QoSProfile(
            depth=1, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        self.sensor_combined_subscription = self.create_subscription(
            SensorCombined, '/fmu/out/sensor_combined', self.sensor_combined_callback, qos)
        self.vehicle_magnetometer_subscription = self.create_subscription(
            VehicleMagnetometer, '/fmu/out/vehicle_magnetometer', self.vehicle_magnetometer_callback, qos)
        self.sensor_mag_subscription = self.create_subscription(
            SensorMag, '/fmu/out/sensor_mag', self.sensor_mag_callback, qos)

    def sensor_combined_callback(self, msg):
        imu_msg = Imu()
        timestamp_microseconds = msg.timestamp
        imu_msg.header.stamp = Time(
            nanoseconds=timestamp_microseconds * 1000).to_msg()
        imu_msg.header.frame_id = 'base_link'
        imu_msg.linear_acceleration.x = float(msg.accelerometer_m_s2[0])
        imu_msg.linear_acceleration.y = float(msg.accelerometer_m_s2[1])
        imu_msg.linear_acceleration.z = float(msg.accelerometer_m_s2[2])
        imu_msg.angular_velocity.x = float(msg.gyro_rad[0])
        imu_msg.angular_velocity.y = float(msg.gyro_rad[1])
        imu_msg.angular_velocity.z = float(msg.gyro_rad[2])
        self.imu_publisher.publish(imu_msg)

    def vehicle_magnetometer_callback(self, msg):
        mag_msg = MagneticField()
        timestamp_microseconds = msg.timestamp
        mag_msg.header.stamp = Time(
            nanoseconds=timestamp_microseconds * 1000).to_msg()
        mag_msg.header.frame_id = 'base_link'
        mag_msg.magnetic_field.x = float(msg.magnetometer_ga[0])
        mag_msg.magnetic_field.y = float(msg.magnetometer_ga[1])
        mag_msg.magnetic_field.z = float(msg.magnetometer_ga[2])
        self.mag_publisher.publish(mag_msg)

    def sensor_mag_callback(self, msg):
        mag_msg = MagneticField()
        timestamp_microseconds = msg.timestamp
        mag_msg.header.stamp = Time(
            nanoseconds=timestamp_microseconds * 1000).to_msg()
        mag_msg.header.frame_id = 'base_link'
        mag_msg.magnetic_field.x = msg.x
        mag_msg.magnetic_field.y = msg.y
        mag_msg.magnetic_field.z = msg.z
        self.mag_publisher.publish(mag_msg)


def main(args=None):
    rclpy.init(args=args)
    sensor_translator = SensorTranslator()
    rclpy.spin(sensor_translator)
    sensor_translator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
