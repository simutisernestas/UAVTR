import os
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import px4_msgs
import pymap3d as pm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import mplcursors


storage_id = 'sqlite3'
SAVE = True
WHICH = 1  # 0 or 1
PRINT_STATIC_ZERO = False


def get_rosbag_options(path, serialization_format='cdr'):
    global storage_id

    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def read_bag(path, clear_cache=False) -> (np.array, np.array):
    CACHE_DIR = '/tmp/rosbag2_py/.cache'
    os.makedirs(CACHE_DIR, exist_ok=True)

    if clear_cache:
        for f in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, f))

    # try load from cache
    bag_cache = CACHE_DIR + '/' + path.split('/')[-1] + '.npz'
    if os.path.isfile(bag_cache):
        print(f"Loading from cache {path.split('/')[-1]}")
        npzfile = np.load(bag_cache)
        return (npzfile['time'], npzfile['position'], npzfile['velocity'])

    storage_options, converter_options = get_rosbag_options(bag_path)

    reader = rosbag2_py.SequentialCompressionReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {  # Create a map for quicker lookup
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

        velocity = np.array([msg.vel_n_m_s, msg.vel_e_m_s, msg.vel_d_m_s])

        lat = msg.lat * 1e-7
        lon = msg.lon * 1e-7
        alt = msg.alt * 1e-3
        if WHICH == 1:
            time_utc = msg.time_utc_usec / 1e6
        else:
            time_utc = msg.timestamp / 1e6
        return (lat, lon, alt, time_utc, velocity)

    (lat0, lon0, alt0) = None, None, None
    if PRINT_STATIC_ZERO:
        out = get_gps_or_time(reader)
        while isinstance(out, px4_msgs.msg.TimesyncStatus):
            out = get_gps_or_time(reader)
        print(out)
        exit(0)
    else:
        lat0, lon0, alt0 = (55.603376, 12.3866865, -3.34)

    buff = []
    timestamps = []
    velocities = []
    # latest_stamp = 0
    while True:
        out = None
        try:
            out = get_gps_or_time(reader)
        except:
            break

        if isinstance(out, px4_msgs.msg.TimesyncStatus):
            # latest_stamp = (out.timestamp - out.observed_offset) / 1e6
            continue

        (lat, lon, alt, tutc, vel) = out
        enu_xyz = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        buff.append(enu_xyz)
        timestamps.append(tutc)
        velocities.append(vel)

    # cache
    np.savez(bag_cache, time=timestamps, position=buff, velocity=velocities)

    return (np.array(timestamps), np.array(buff), np.array(velocities).reshape(-1, 3))


if __name__ == '__main__':
    # parent directory
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(scripts_dir)

    if WHICH == 0:
        bag_path = f"{root_dir}/bags/18_0/rosbag2_2023_10_18-12_24_19"
    else:
        bag_path = f"{root_dir}/bags/latest_flight/rosbag2_2023_10_18-16_22_16"

    (timestamps, buff, vel) = read_bag(bag_path, clear_cache=True)

    if WHICH == 0:
        bag_path = f"{root_dir}/bags/18_0/rosbag2_2023_10_18-12_15_37"
    else:
        bag_path = f"{root_dir}/bags/latest_flight/rosbag2_2023_08_21-23_15_45"

    (timestamps_boat, buff_boat, _) = read_bag(bag_path, clear_cache=True)

    if SAVE:
        save_file = scripts_dir + f"/data/{bag_path.split('/')[-2]}_gt.npz"
        print(f"Saving to {save_file}")
        # cache these GT values in one file
        np.savez(save_file,
                 drone_time=timestamps,
                 boat_time=timestamps_boat,
                 drone_pos=buff,
                 boat_pos=buff_boat,
                 drone_vel=vel)

    colors = cm.rainbow(np.linspace(0, 1, len(buff)))
    pobj = plt.scatter([x[0] for x in buff], [x[1] for x in buff], color='b')

    cursor = mplcursors.cursor(pobj, hover=True)
    cursor.connect(
        "add",
        lambda sel: sel.annotation.set_text(
            f'Time: {timestamps[sel.target.index]}')
    )

    colors = cm.rainbow(np.linspace(0, 1, len(buff_boat)))
    pobj = plt.scatter([x[0] for x in buff_boat], [x[1]
                                                   for x in buff_boat], color='r')

    cursor = mplcursors.cursor(pobj, hover=True)
    cursor.connect(
        "add",
        lambda sel: sel.annotation.set_text(
            f'Time: {timestamps_boat[sel.target.index]}')
    )
    plt.show()
