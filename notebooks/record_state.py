import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
import numpy as np
import sys
import os
import time

class StateSubscriber(Node):
    def __init__(self, bag_name):
        super().__init__('state_subscriber')
        self.data = []
        self.counter = 0
        self.subscription = self.create_subscription(
            PoseArray,
            '/state',
            self.callback,
            100)
        self.subscription  # prevent unused variable warning
        timestamp = int(time.time())
        self.data_file = f'data/{timestamp}_{bag_name}_state_data.npy'
        # remove the file before starting
        try:
            os.remove(self.data_file)
        except FileNotFoundError:
            pass

    def callback(self, msg):
        timestamp = msg.header.stamp
        unix_timestamp = timestamp.sec + timestamp.nanosec * 1e-9
        self.data.append(unix_timestamp)
        for pose in msg.poses:
            self.data.extend([pose.position.x, pose.position.y, pose.position.z])
        self.data.append(msg.poses[0].orientation.x) # target in sight?
        self.counter += 1
        if self.counter % 1000 == 0:  # save every nth messages
            self.save_data()

    def save_data(self):
        np_data = np.array(self.data)
        try: # load and append
            np_data = np.load(self.data_file)
            np_data = np.append(np_data, self.data)
        except FileNotFoundError:
            pass
        np.save(self.data_file, np_data)
        self.data = []  # clear the list

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