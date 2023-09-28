#!/usr/bin/env python3

# /camera/accel/sample [sensor_msgs/msg/Imu]
# /camera/color/camera_info [sensor_msgs/msg/CameraInfo]
# /camera/color/image_raw [sensor_msgs/msg/Image]
# /camera/gyro/sample [sensor_msgs/msg/Imu]
# /camera/imu [sensor_msgs/msg/Imu]
# /fmu/out/sensor_baro [px4_msgs/msg/SensorBaro]
# /fmu/out/sensor_combined [px4_msgs/msg/SensorCombined]
# /fmu/out/sensor_mag [px4_msgs/msg/SensorMag]
# /fmu/out/vehicle_attitude [px4_msgs/msg/VehicleAttitude]
# /fmu/out/vehicle_global_position [px4_msgs/msg/VehicleGlobalPosition]
# /fmu/out/vehicle_local_position [px4_msgs/msg/VehicleLocalPosition]
# /fmu/out/vehicle_odometry [px4_msgs/msg/VehicleOdometry]
# /fmu/out/vehicle_gps_position [px4_msgs/msg/SensorGps]
# /fmu/out/vehicle_magnetometer [px4_msgs/msg/VehicleMagnetometer]
# /tf_static [tf2_msgs/msg/TFMessage]

import unittest
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, CameraInfo, Image, Range
from px4_msgs.msg import TimesyncStatus, SensorBaro, SensorCombined, SensorMag, VehicleAttitude, VehicleGlobalPosition, VehicleLocalPosition, VehicleOdometry, SensorGps, VehicleMagnetometer
from tf2_msgs.msg import TFMessage
from parameterized import parameterized

N_SPIN = 200

class TestTopicReception(unittest.TestCase):

    def callback(self, msg):
        self.time_msg = msg

    def setUp(self):
        rclpy.init()
        self.node = Node("test_node")
        self.time_msg = None
        qos = rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        self.time_sub = self.node.create_subscription(TimesyncStatus, "/fmu/out/timesync_status", self.callback, qos)

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_timesync_reception(self):
        for _ in range(N_SPIN):
            if self.time_msg is not None:
                break
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.assertIsNotNone(self.time_msg)
        self.assertTrue(self.time_msg.observed_offset != 0)

    @parameterized.expand([
        ["/camera/accel/sample", Imu],
        ["/camera/gyro/sample", Imu],
        ["/camera/imu", Imu],
        ["/camera/color/camera_info", CameraInfo],
        ["/camera/color/image_raw", Image],
        ["/fmu/out/sensor_baro", SensorBaro],
        ["/fmu/out/sensor_combined", SensorCombined],
        ["/fmu/out/sensor_mag", SensorMag],
        ["/fmu/out/vehicle_attitude", VehicleAttitude],
        ["/fmu/out/vehicle_global_position", VehicleGlobalPosition],
        ["/fmu/out/vehicle_local_position", VehicleLocalPosition],
        ["/fmu/out/vehicle_odometry", VehicleOdometry],
        ["/fmu/out/vehicle_gps_position", SensorGps],
        ["/fmu/out/vehicle_magnetometer", VehicleMagnetometer],
        ["/teraranger_evo_40m", Range]
    ])        
    def test_topic_reception_and_timestamp(self, topic, msg_type, check_timestamp=True):
        received = False
        timestamp = None
        def callback(msg):
            nonlocal received
            nonlocal timestamp
            if self.time_msg is None:
                return
            self.assertIsInstance(msg, msg_type)
            received = True
            try:
                timestamp = rclpy.time.Time.from_msg(msg.header.stamp)
            except Exception as e:
                # px4 timestamp is in microseconds and has offset compared to unix time
                offset = 0 if self.time_msg.estimated_offset != 0 else self.time_msg.observed_offset*1e3
                timestamp = rclpy.time.Time(
                    nanoseconds=msg.timestamp*1e3 - offset,
                    clock_type=self.node.get_clock().clock_type)

        # create sub
        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        sub = self.node.create_subscription(msg_type, topic, callback, qos)
        
        # receive messages
        for _ in range(N_SPIN):
            if received and self.time_msg is not None:
                break
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertTrue(received)
        self.node.destroy_subscription(sub)
        
        # check timestamp
        if not check_timestamp:
            return
        current_time = self.node.get_clock().now()
        diff = current_time - timestamp
        error = f"Diff: {diff.nanoseconds}, current: {current_time.nanoseconds}, msg_ts: {timestamp.nanoseconds}"
        self.assertTrue(diff.nanoseconds > 0, error)
        self.assertTrue(diff.nanoseconds < 1e9, error)

    def test_transforms(self):
        received = False
        tf_msg = []
        def callback(msg):
            nonlocal received
            nonlocal tf_msg
            self.assertIsInstance(msg, TFMessage)
            received = len(tf_msg) > 0
            tf_msg.append(msg)

        # create sub
        qos = rclpy.qos.QoSProfile(depth=10, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        sub = self.node.create_subscription(TFMessage, "/tf_static", callback, qos)

        # receive messages
        for _ in range(N_SPIN):
            if received:
                break
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertTrue(received)
        self.node.destroy_subscription(sub)
        
        frames = []
        for tf_array in tf_msg:
            for tf in tf_array.transforms:
                if tf.child_frame_id in frames:
                    continue
                frames.append(tf.child_frame_id)

        # expected frames
        expected_frames = [
            "camera_color_frame",
            "camera_color_optical_frame",
            "camera_accel_frame",
            "camera_accel_optical_frame",
            "camera_gyro_frame",
            "camera_gyro_optical_frame",
            "camera_imu_frame",
            "camera_imu_optical_frame",
            "camera_link",
            "gps_link"
        ]
        self.assertCountEqual(frames, expected_frames, frames)

if __name__ == '__main__':
    unittest.main()