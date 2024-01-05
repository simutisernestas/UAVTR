from collections import deque
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os

def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


class LowPassFilter:
    def __init__(self, b_coefficients, a_coefficients):
        self.b_coefficients = b_coefficients
        self.a_coefficients = a_coefficients
        self.input_samples = deque([0.0] * len(b_coefficients))
        self.filter_buffer = deque([0.0] * (len(b_coefficients) - 1))

    def filter(self, input):
        assert len(self.input_samples) == len(self.b_coefficients)
        assert len(self.filter_buffer) == len(self.a_coefficients) - 1

        output = 0.0

        # push the new sample into the input samples buffer
        self.input_samples.appendleft(input)
        self.input_samples.pop()

        # compute the new output
        for i in range(len(self.filter_buffer)):
            output -= self.a_coefficients[i + 1] * self.filter_buffer[i]
        for i in range(len(self.input_samples)):
            output += self.b_coefficients[i] * self.input_samples[i]

        # push the new output into the filter buffer
        self.filter_buffer.appendleft(output)
        self.filter_buffer.pop()

        return output


storage_id = 'sqlite3'

if __name__ == '__main__':
    PANDA_MSGS = []
    LAPTOP_MSGS = []
    N = 5
    
    root_dir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))) # parent; project root

    bag_path = f"{root_dir}/bags/18_0/rosbag2_2023_10_18-12_24_19"

    info = rosbag2_py.Info()
    metadata = info.read_metadata(bag_path, storage_id)
    panda_start_timestamp = metadata.starting_time.timestamp()

    storage_options, converter_options = get_rosbag_options(
        bag_path, storage_id)

    reader = rosbag2_py.SequentialCompressionReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()

    # Create a map for quicker lookup
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    # Set filter for topic of string type
    storage_filter = rosbag2_py.StorageFilter(
        topics=['/fmu/out/sensor_combined'])
    reader.set_filter(storage_filter)

    N = 1000
    imu_data = np.zeros((1000, 3))

    msg_counter = 0
    while reader.has_next():
        if msg_counter < 160*200:
            msg_counter += 1
            (topic, data, t) = reader.read_next()
            continue
        break

    msg_counter = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        imu_data[msg_counter] = msg.accelerometer_m_s2
        msg_counter += 1
        if msg_counter == N:
            break

    # plot imu data
    plt.plot(imu_data[:, 0], label='x')
    # plt.plot(imu_data[:, 1], label='y')
    # plt.plot(imu_data[:, 2], label='z')

    # Design the low pass filter.
    filter_order = 2
    cutoff_frequency = 10
    b, a = butter(filter_order, cutoff_frequency,
                  fs=200, btype="lowpass", analog=False)
    
    iir = LowPassFilter(b, a)
    print(a)
    print(b)

    # Filter the IMU accelerometer data using the low pass filter.
    filtered_imu_data = np.zeros(N)
    for i in range(N):
        filtered_imu_data[i] = iir.filter(imu_data[i,0])
    
    # plot filtered imu data
    plt.plot(filtered_imu_data, label='x filtered')
    # exit()

    # Filter the IMU accelerometer data using the low pass filter.
    filtered_imu_data = lfilter(b, a, imu_data[:, 0])

    print(filtered_imu_data.shape)
    print(np.mean(filtered_imu_data))
    print(np.median(filtered_imu_data))

    # plot filtered imu data
    plt.plot(filtered_imu_data, label='x filtered scipy')
    plt.legend()
    plt.show()

    # # export the imu data to text file values separated by comma, also add coefficients a & b to the end of the file
    # np.savetxt("imu_data.csv", imu_data, delimiter=",")
    # with open("imu_data.csv", "a") as f:
    #     # f.write(f"\n{list(a)}\n{b}")
    #     for i in range(len(a)):
    #         f.write(f"{a[i]},")
    #     f.write("\n")
    #     for i in range(len(b)):
    #         f.write(f"{b[i]},")

    # # print average of filtered imu data


    # print(filtered_imu_data)
