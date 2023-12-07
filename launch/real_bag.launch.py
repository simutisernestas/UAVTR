from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os


def generate_launch_description():
    root_dir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))

    orientation_filter = ExecuteProcess(
        cmd=[f'{root_dir}/src/estimation/build/orientation_filter'],
        output='screen'
    )

    tracking = ExecuteProcess(
        cmd=['./tracking_ros_node'],
        cwd=f'{root_dir}/src/detection/build',
        output='screen'
    )

    estimation = ExecuteProcess(
        cmd=["./estimation_node"],
        cwd=f'{root_dir}/src/estimation/build',
        # prefix=['xterm  -e gdb -ex "b main" --args'],
        output='screen'
    )

    uncompress = ExecuteProcess(
        cmd=['ros2', 'run', 'image_transport', 'republish', 'compressed', 'raw', '--ros-args', '-r',
             '/in/compressed:=/camera/color/image_raw/compressed', '-r', 'out:=/camera/color/image_raw'],
        output='screen'
    )

    bag_name = "./18_0/rosbag2_2023_10_18-12_24_19"
    # bag_name = "./latest_flight/rosbag2_2023_10_18-16_22_16/"
    play_bag_cmd = f'''ros2 bag play {bag_name} --start-offset 150'''
    play_bag = ExecuteProcess(
        cmd=play_bag_cmd.split(),
        cwd=f"{root_dir}/bags",
        output='screen'
    )

    record_state = ExecuteProcess(
        cmd=["python3", "record_state.py", bag_name.split('/')[-2]],
        cwd=f"{root_dir}/notebooks",
        output='screen'
    )

    return LaunchDescription([
        play_bag,
        tracking,
        estimation,
        uncompress,
        orientation_filter,
        record_state
    ])
