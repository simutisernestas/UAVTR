from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os


def generate_launch_description():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    imu_mag_repub = ExecuteProcess(
        cmd=['/home/ernie/thesis/track/src/estimation/build/orientation_filter'],
        output='screen'
    )

    tracking = ExecuteProcess(
        cmd=['./tracking_ros_node'],
        cwd='/home/ernie/thesis/track/src/detection/build',
        output='screen'
    )

    estimation = ExecuteProcess(
        cmd=["./estimation_node"],
        cwd='/home/ernie/thesis/track/src/estimation/build',
        # prefix=['xterm -fa "Monospace" -fs 14 -e gdb --args'],
        # prefix=['xterm  -e gdb -ex "b main" --args'],
        # -iex break -ex "b main" -tui 
        output='screen'
    )

    uncompress = ExecuteProcess(
        cmd=['ros2', 'run', 'image_transport', 'republish', 'compressed', 'raw', '--ros-args', '-r',
             '/in/compressed:=/camera/color/image_raw/compressed', '-r', 'out:=/camera/color/image_raw'],
        output='screen'
    )

    play_bag_cmd = '''ros2 bag play ./18_0/rosbag2_2023_10_18-12_24_19 --start-offset 150 -l'''
    play_bag = ExecuteProcess(
        cmd=play_bag_cmd.split(),
        cwd="/home/ernie/thesis/bags",
        output='screen'
    )

    return LaunchDescription([
        play_bag,
        tracking,
        # estimation,
        uncompress,
        imu_mag_repub,
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([
        #         PathJoinSubstitution([
        #             dir_path,
        #             "madgwick.launch.py"
        #         ])
        #     ]),
        # ),
    ])
