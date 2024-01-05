import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import Imu
import numpy as np
import sys
import os
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

        timestamp = int(time.time())
        self.pos_data_file = f'data/{timestamp}_{bag_name}_state_data.npy'
        self.attitude_state_file = f'data/{timestamp}_{bag_name}_attitude_state.npy'
        self.attitude_px4_file = f'data/{timestamp}_{bag_name}_attitude_px4.npy'

    def state_callback(self, msg):
        timestamp = msg.header.stamp
        unix_timestamp = timestamp.sec + timestamp.nanosec * 1e-9
        self.state_data.append(unix_timestamp)
        for pose in msg.poses:
            self.state_data.extend(
                [pose.position.x, pose.position.y, pose.position.z])
        self.state_data.append(msg.poses[0].orientation.x)  # target in sight?
        self.state_data.append(msg.poses[1].orientation.x)  # Cov X
        self.state_data.append(msg.poses[2].orientation.x)  # Cov Y
        self.state_data.append(msg.poses[3].orientation.x)  # Cov Z
        self.state_counter += 1
        if self.state_counter % 100 == 0:  # save every nth messages
            self.save_data('state')

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
