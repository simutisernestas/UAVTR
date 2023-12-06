import os
from pathlib import Path
import pytest
from rcl_interfaces.msg import Log
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String

import px4_msgs
import plotly.express as px
import pandas as pd
import pymap3d as pm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import mplcursors


storage_id = 'sqlite3'

def get_rosbag_options(path, serialization_format='cdr'):
    global storage_id

    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options



if __name__ == '__main__':
    # parent directory
    root_dir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))

    # bag_path = f"{root_dir}/bags/latest_flight/rosbag2_2023_10_18-16_22_16"
    bag_path = f"{root_dir}/bags/18_0/rosbag2_2023_10_18-12_24_19"

    info = rosbag2_py.Info()
    metadata = info.read_metadata(bag_path, storage_id)
    panda_start_timestamp = metadata.starting_time.timestamp()

    storage_options, converter_options = get_rosbag_options(bag_path)

    reader = rosbag2_py.SequentialCompressionReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = { # Create a map for quicker lookup
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    storage_filter = rosbag2_py.StorageFilter(
        topics=[
            '/fmu/out/vehicle_gps_position',
            '/fmu/out/timesync_status',
        ])
    reader.set_filter(storage_filter)

    def get_gps_or_time(reader):
        if not reader.has_next():
            raise Exception("No more msgs")

        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        if not isinstance(msg, px4_msgs.msg.SensorGps):
            return msg

        lat = msg.lat * 1e-7
        lon = msg.lon * 1e-7
        alt = msg.alt * 1e-3
        time_utc = msg.timestamp / 1e6
        return (lat, lon, alt, time_utc)

    (lat0, lon0, alt0) = None, None, None
    out = get_gps_or_time(reader)
    while isinstance(out, px4_msgs.msg.TimesyncStatus):
        out = get_gps_or_time(reader)
    (lat0, lon0, alt0, tutc) = out

    buff = []
    timestamps = []
    latest_stamp = 0
    while True:
        out = None
        try:
            out = get_gps_or_time(reader)
        except:
            break

        if isinstance(out, px4_msgs.msg.TimesyncStatus):
            latest_stamp = (out.timestamp - out.observed_offset) / 1e6
            continue

        (lat, lon, alt, tutc) = out
        enu_xyz = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        buff.append(enu_xyz)
        timestamps.append(tutc)

    # bag_path = f"{root_dir}/bags/latest_flight/rosbag2_2023_08_21-23_15_45"
    bag_path = f"{root_dir}/bags/18_0/rosbag2_2023_10_18-12_15_37"

    info = rosbag2_py.Info()
    metadata = info.read_metadata(bag_path, storage_id)
    panda_start_timestamp = metadata.starting_time.timestamp()

    storage_options, converter_options = get_rosbag_options(bag_path)

    reader = rosbag2_py.SequentialCompressionReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    reader.set_filter(storage_filter)

    buff_boat = []
    timestamps_boat = []
    latest_stamp = 0
    while True:
        out = None
        try:
            out = get_gps_or_time(reader)
        except:
            break

        if isinstance(out, px4_msgs.msg.TimesyncStatus):
            # DELTA_S = 4986634
            #  / 1e6 + DELTA_S
            latest_stamp = (out.timestamp - out.observed_offset) / 1e6
            continue

        (lat, lon, alt, tutc) = out
        enu_xyz = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        buff_boat.append(enu_xyz)
        timestamps_boat.append(tutc)

    timestamps = np.array(timestamps)
    timestamps_boat = np.array(timestamps_boat)
    buff = np.array(buff)
    buff_boat = np.array(buff_boat)
    # cache these GT values in one file
    np.savez(bag_path.split('/')[-2] + '_gt.npz', 
             drone_time=timestamps, 
             boat_time=timestamps_boat, 
             drone_pos=buff, 
             boat_pos=buff_boat)

    colors = cm.rainbow(np.linspace(0, 1, len(buff)))
    pobj = plt.scatter([x[0] for x in buff], [x[1] for x in buff], color='b')
    cursor = mplcursors.cursor(pobj, hover=True)
    cursor.connect(
        "add", 
        lambda sel: sel.annotation.set_text(f'Time: {timestamps[sel.target.index]}')
    )

    colors = cm.rainbow(np.linspace(0, 1, len(buff_boat)))
    pobj = plt.scatter([x[0] for x in buff_boat], [x[1]
                for x in buff_boat], color='r')

    cursor = mplcursors.cursor(pobj, hover=True)
    cursor.connect(
        "add", 
        lambda sel: sel.annotation.set_text(f'Time: {timestamps_boat[sel.target.index]}')
    )
    plt.show()