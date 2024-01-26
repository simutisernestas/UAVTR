import rclpy
from rclpy.node import Node
from geometry_msgs.msg import (
    PoseArray, PointStamped, TwistStamped)
from sensor_msgs.msg import Imu
import numpy as np
import sys
import time


class StateSubscriber(Node):
    def __init__(self, bag_name):
        super().__init__('state_subscriber')
        self.state_data = []
        self.state_counter = 0
        self.state_subscription = self.create_subscription(
            PoseArray,
            '/state',
            self.state_callback,
            1000)
        self.attitude_est_data = []
        self.attitude_est_counter = 0
        self.subscription_imu_euler = self.create_subscription(
            Imu,
            '/imu/euler',
            self.callback_imu,
            1000)
        self.attitude_px4_data = []
        self.attitude_px4_counter = 0
        self.subscription_imu_euler_px4 = self.create_subscription(
            Imu,
            '/imu/euler_px4',
            self.callback_imu_px4,
            1000)
        self.imu_data = []
        self.imu_counter = 0
        self.subscription_imu = self.create_subscription(
            Imu,
            '/imu/data_raw',
            self.callback_imu_raw,
            1000)
        self.vel_measurements = []
        self.vel_measurements_counter = 0
        self.subscription_vel_measurements = self.create_subscription(
            TwistStamped,
            '/vel',
            self.callback_vel_measurements,
            1000)
        self.pos_measurements = []
        self.pos_measurements_counter = 0
        self.subscription_pos_measurements = self.create_subscription(
            PointStamped,
            '/target_point',
            self.callback_pos_measurements,
            1000)

        timestamp = int(time.time())
        self.pos_data_file = f'data/{timestamp}_{bag_name}_state_data.npy'
        self.attitude_state_file = f'data/{timestamp}_{bag_name}_attitude_state.npy'
        self.attitude_px4_file = f'data/{timestamp}_{bag_name}_attitude_px4.npy'
        self.vel_measurements_file = f'data/{timestamp}_{bag_name}_vel_measurements.npy'
        self.pos_measurements_file = f'data/{timestamp}_{bag_name}_pos_measurements.npy'
        self.imu_data_file = f'data/{timestamp}_{bag_name}_imu_data.npy'

    def state_callback(self, msg):
        timestamp = msg.header.stamp
        unix_timestamp = timestamp.sec + timestamp.nanosec * 1e-9
        self.state_data.append(unix_timestamp)
        for pose in msg.poses:
            self.state_data.extend(
                [pose.position.x, pose.position.y, pose.position.z])
        self.state_data.append(msg.poses[0].orientation.x)  # Target in sight?
        self.state_data.append(msg.poses[1].orientation.x)  # Pos Cov X
        self.state_data.append(msg.poses[2].orientation.x)  # Pos Cov Y
        self.state_data.append(msg.poses[3].orientation.x)  # Pos Cov Z
        self.state_data.append(msg.poses[1].orientation.y)  # Vel Drone Cov X
        self.state_data.append(msg.poses[2].orientation.y)  # Vel Drone Cov Y
        self.state_data.append(msg.poses[3].orientation.y)  # Vel Drone Cov Z
        self.state_data.append(msg.poses[1].orientation.z)  # Vel Boat Cov X
        self.state_data.append(msg.poses[2].orientation.z)  # Vel Boat Cov Y
        self.state_counter += 1
        if self.state_counter % 100 == 0:  # save every nth messages
            self.save_data('state')

    def callback_imu_raw(self, msg):
        timestamp = msg.header.stamp
        unix_timestamp = timestamp.sec + timestamp.nanosec * 1e-9
        self.imu_data.append(unix_timestamp)
        self.imu_data.extend(
            [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.imu_counter += 1
        if self.imu_counter % 100 == 0:
            self.save_data('imu')

    def callback_imu(self, msg):
        timestamp = msg.header.stamp
        unix_timestamp = timestamp.sec + timestamp.nanosec * 1e-9
        self.attitude_est_data.append(unix_timestamp)
        self.attitude_est_data.extend(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.attitude_est_counter += 1
        if self.attitude_est_counter % 100 == 0:  # save every nth messages
            self.save_data('attitude_est')

    def callback_imu_px4(self, msg):
        timestamp = msg.header.stamp
        unix_timestamp = timestamp.sec + timestamp.nanosec * 1e-9
        self.attitude_px4_data.append(unix_timestamp)
        self.attitude_px4_data.extend(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.attitude_px4_counter += 1
        if self.attitude_px4_counter % 30 == 0:  # save every nth messages
            self.save_data('attitude_px4')

    def callback_vel_measurements(self, msg):
        timestamp = msg.header.stamp
        unix_timestamp = timestamp.sec + timestamp.nanosec * 1e-9
        self.vel_measurements.append(unix_timestamp)
        self.vel_measurements.extend(
            [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.vel_measurements_counter += 1
        if self.vel_measurements_counter % 100 == 0:
            self.save_data('vel_measurements')

    def callback_pos_measurements(self, msg):
        timestamp = msg.header.stamp
        unix_timestamp = timestamp.sec + timestamp.nanosec * 1e-9
        self.pos_measurements.append(unix_timestamp)
        self.pos_measurements.extend(
            [msg.point.x, msg.point.y, msg.point.z])
        self.pos_measurements_counter += 1
        if self.pos_measurements_counter % 100 == 0:
            self.save_data('pos_measurements')

    def save_data(self, which='state'):
        if which == 'state':
            data_buff = self.state_data
            data_file = self.pos_data_file
        elif which == 'attitude_est':
            data_buff = self.attitude_est_data
            data_file = self.attitude_state_file
        elif which == 'attitude_px4':
            data_buff = self.attitude_px4_data
            data_file = self.attitude_px4_file
        elif which == 'vel_measurements':
            data_buff = self.vel_measurements
            data_file = self.vel_measurements_file
        elif which == 'pos_measurements':
            data_buff = self.pos_measurements
            data_file = self.pos_measurements_file
        elif which == 'imu':
            data_buff = self.imu_data
            data_file = self.imu_data_file
        else:
            raise ValueError(
                'which must be one of state, attitude_est or attitude_px4')

        np_data = np.array(data_buff)
        try:  # load and append
            np_data = np.load(data_file)
            np_data = np.append(np_data, data_buff)
        except FileNotFoundError:
            pass
        np.save(data_file, np_data)

        if which == 'state':
            self.state_data = []
        elif which == 'attitude_est':
            self.attitude_est_data = []
        elif which == 'attitude_px4':
            self.attitude_px4_data = []
        elif which == 'vel_measurements':
            self.vel_measurements = []
        elif which == 'pos_measurements':
            self.pos_measurements = []
        elif which == 'imu':
            self.imu_data = []


def main():
    # take first argument from command line as bag name
    bag_name = sys.argv[1]
    rclpy.init()
    state_subscriber = StateSubscriber(bag_name)
    rclpy.spin(state_subscriber)
    state_subscriber.save_data()  # save remaining data when node is shut down
    state_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
