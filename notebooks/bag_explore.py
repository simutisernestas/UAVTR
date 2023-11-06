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


def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


storage_id = 'sqlite3'
PLOT_GPS = False

if __name__ == '__main__':
    PANDA_MSGS = []
    LAPTOP_MSGS = []
    N = 5

    # bag_path = "/home/ernie/thesis/rosbag2_2023_11_02-14_25_42"
    # info = rosbag2_py.Info()
    # metadata = info.read_metadata(bag_path, storage_id)
    # panda_start_timestamp = metadata.starting_time.timestamp()

    # storage_options, converter_options = get_rosbag_options(
    #     bag_path, storage_id)

    # reader = rosbag2_py.SequentialReader()
    # reader.open(storage_options, converter_options)

    # topic_types = reader.get_all_topics_and_types()

    # # Create a map for quicker lookup
    # type_map = {
    #     topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    # # Set filter for topic of string type
    # storage_filter = rosbag2_py.StorageFilter(
    #     topics=['/cam_target_pos'])
    # reader.set_filter(storage_filter)

    # cam_pos_buff = []
    # while reader.has_next():
    #     (topic, data, t) = reader.read_next()
    #     msg_type = get_message(type_map[topic])
    #     msg = deserialize_message(data, msg_type)
    #     x, y, z = msg.point.x, msg.point.y, msg.point.z
    #     if abs(x) > 40 or abs(y) > 40:
    #         continue
    #     if np.linalg.norm([x, y, z]) < 10:
    #         continue
    #     cam_pos_buff.append([msg.point.x, msg.point.y, msg.point.z])

    # norm_of_each_point = [np.linalg.norm(x) for x in cam_pos_buff[:13000]]

    # plt.plot(norm_of_each_point)

    # # colors = cm.rainbow(np.linspace(0, 1, len(cam_pos_buff)))
    # # plt.scatter([x[0] for x in cam_pos_buff], [x[1] for x in cam_pos_buff], color=colors)
    # plt.show()
    # exit()

    # drwxrwxr-x 2 ernie ernie 4.0K Oct 18 12:31 rosbag2_2023_10_18-12_15_37
    # drwxrwxr-x 2 ernie ernie 4.0K Oct 18 12:30 rosbag2_2023_10_18-12_24_19
    bag_path = "/home/ernie/thesis/bags/18_0/rosbag2_2023_10_18-12_24_19"

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
        topics=['/fmu/out/vehicle_gps_position'])
    reader.set_filter(storage_filter)

    def get_gps_pos(reader):
        if not reader.has_next():
            raise Exception("No gps msgs")

        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        if not isinstance(msg, px4_msgs.msg.SensorGps):
            raise Exception("Not a gps msg")

        lat = msg.lat * 1e-7
        lon = msg.lon * 1e-7
        alt = msg.alt * 1e-3
        return (lat, lon, alt)

    (lat0, lon0, alt0) = get_gps_pos(reader)

    buff = []
    while True:
        try:
            (lat, lon, alt) = get_gps_pos(reader)
        except:
            break
        enu_xyz = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        buff.append(enu_xyz)

    bag_path = "/home/ernie/thesis/bags/18_0/rosbag2_2023_10_18-12_15_37"

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
        topics=['/fmu/out/vehicle_gps_position'])
    reader.set_filter(storage_filter)

    buff_boat = []
    count = 0
    while True:
        try:
            (lat, lon, alt) = get_gps_pos(reader)
        except:
            break
        count += 1
        if count < 0:
            continue
        if count == int(4800//2):
            print(lat, lon, alt)
            exit()
        enu_xyz = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        buff_boat.append(enu_xyz)
        # break

    # Create a color map
    colors = cm.rainbow(np.linspace(0, 1, len(buff)))
    plt.scatter([x[0] for x in buff], [x[1] for x in buff], color=colors)
    colors = cm.rainbow(np.linspace(0, 1, len(buff_boat)))
    # dump buff_boat to file
    with open("buff_boat.txt", "w") as f:
        for i in range(len(buff_boat)):
            f.write(f"{buff_boat[i][0]} {buff_boat[i][1]} {buff_boat[i][2]}\n")
    from_nth_point_black = []
    for i in range(len(colors)):
        if i == int(4800//2):
            color = colors[i]
            color = (0, 0, 0, 1)            
            from_nth_point_black.append(color)
        else:
            # make alpha low
            color = colors[i]
            color = (color[0], color[1], color[2], 0.01)
            from_nth_point_black.append(color)
    plt.scatter([x[0] for x in buff_boat], [x[1]
                for x in buff_boat], color=from_nth_point_black)
    # plt.figure()
    # # plot the norm of each buff point to the boat point 0
    # norm_of_each_point = []
    # for i in range(len(buff)):
    #     if i < 500 or i > 1750:
    #         continue
    #     norm_of_each_point.append(np.linalg.norm(
    #         np.array(buff[i]) - np.array(buff_boat[0])))
    # plt.plot(norm_of_each_point)
    plt.show()

    # while reader.has_next():
    #     (topic, data, t) = reader.read_next()
    #     msg_type = get_message(type_map[topic])
    #     msg = deserialize_message(data, msg_type)
    #     lat = msg.lat * 1e-7
    #     lon = msg.lon * 1e-7
    #     alt = msg.alt * 1e-3
    #     enu = pm.geodetic2enu(lat, lon, alt, 0, 0, 0)
    #     break
    exit()

    # if not PLOT_GPS:
    #     # TODO: add this
    #     # , '/fmu/out/timesync_status'
    #     pass

    # msg_counter = 0
    # while reader.has_next():
    #     (topic, data, t) = reader.read_next()
    #     msg_type = get_message(type_map[topic])
    #     msg = deserialize_message(data, msg_type)

    #     # print(msg)
    #     # print()
    #     PANDA_MSGS.append(msg)

    #     # assert isinstance(msg, String)
    #     # assert msg.data == f'Hello, world! {msg_counter}'

    #     # msg_counter += 1
    #     # if msg_counter == N*2:
    #     #     break

    # if PLOT_GPS:
    #     msg = PANDA_MSGS[0]
    #     print(msg.lat * 1e-7)
    #     print(msg.lon * 1e-7)

    #     # create df from panda msgs
    #     df = pd.DataFrame(columns=['ID', 'Lat', 'Long'])
    #     lats = [msg.lat * 1e-7 for msg in PANDA_MSGS]
    #     longs = [msg.lon * 1e-7 for msg in PANDA_MSGS]
    #     df['ID'] = range(len(lats))
    #     df['Lat'] = lats
    #     df['Long'] = longs
    #     print(df)

    #     color_scale = [(0, 'orange'), (1, 'red')]

    #     fig = px.scatter_mapbox(df,
    #                             lat="Lat",
    #                             lon="Long",
    #                             hover_name="ID",
    #                             color_discrete_sequence=["red"],
    #                             zoom=16,
    #                             height=800,
    #                             width=800)
    #     # adding second trace to the figure
    #     # fig2 = px.line_mapbox(df, lat="lat", lon="lng", zoom=8)
    #     # fig.add_trace(fig2.data[0])  # adds the line trace to the first figure
    #     fig.update_layout(mapbox_style="open-street-map")
    #     fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    #     fig.show()
    #     exit()

    # bag_path = "/home/ernie/thesis/bags/latest-niceish-very-far"
    # storage_options, converter_options = get_rosbag_options(
    #     bag_path, storage_id)

    # metadata = info.read_metadata(bag_path, storage_id)
    # laptop_start_timestamp = metadata.starting_time.timestamp()

    # reader = rosbag2_py.SequentialCompressionReader()
    # reader.open(storage_options, converter_options)

    # topic_types = reader.get_all_topics_and_types()

    # # Create a map for quicker lookup
    # type_map = {
    #     topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    # print(type_map)
    # print()

    # # Set filter for topic of string type
    # storage_filter = rosbag2_py.StorageFilter(
    #     topics=['/fmu/out/vehicle_gps_position', '/fmu/out/timesync_status'])
    # reader.set_filter(storage_filter)

    # msg_counter = 0
    # while reader.has_next():
    #     (topic, data, t) = reader.read_next()
    #     msg_type = get_message(type_map[topic])
    #     msg = deserialize_message(data, msg_type)

    #     print(msg)
    #     print()
    #     LAPTOP_MSGS.append(msg)

    #     # assert isinstance(msg, String)
    #     # assert msg.data == f'Hello, world! {msg_counter}'

    #     msg_counter += 1
    #     if msg_counter == N*2:
    #         break

    # print([x.timestamp for x in PANDA_MSGS])

    # timesync_panda = None
    # # find timesync msg
    # for msg in PANDA_MSGS:
    #     if isinstance(msg, px4_msgs.msg.TimesyncStatus):
    #         timesync_panda = msg
    #         break
    # timesync_laptop = None
    # # find timesync msg
    # for msg in LAPTOP_MSGS:
    #     if isinstance(msg, px4_msgs.msg.TimesyncStatus):
    #         timesync_laptop = msg
    #         break
    # for i in range(N*2):
    #     if not isinstance(PANDA_MSGS[i], px4_msgs.msg.SensorGps):
    #         continue
    #     if not isinstance(LAPTOP_MSGS[i], px4_msgs.msg.SensorGps):
    #         continue
    #     gps_panda = PANDA_MSGS[i]
    #     gps_laptop = LAPTOP_MSGS[i]

    #     print()
    #     print(gps_panda)
    #     print(timesync_panda)
    #     print()
    #     print(gps_laptop)
    #     print(timesync_laptop)

    #     print()
    #     gps_panda.timestamp -= timesync_panda.observed_offset
    #     print(gps_panda.timestamp / 1e6)
    #     print(gps_laptop.timestamp / 1e6)
    #     print((gps_panda.timestamp / 1e6 - gps_laptop.timestamp / 1e6) / 60)

    #     break

    #     print()
    #     print(gps_panda.timestamp)
    #     print(gps_laptop.timestamp - timesync_laptop.observed_offset)

    #     # diff between
    #     print()
    #     diff = gps_panda.timestamp - \
    #         (gps_laptop.timestamp - timesync_laptop.observed_offset)
    #     diff /= 1e6
    #     print(diff)
    #     print(panda_start_timestamp - laptop_start_timestamp)
    #     break
