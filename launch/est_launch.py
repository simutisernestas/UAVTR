from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os


def generate_launch_description():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    imu_mag_repub = ExecuteProcess(
        cmd=['python3', '/home/ernie/thesis/track/repub.py'],
        output='screen'
    )

    tracking = ExecuteProcess(
        cmd=['./tracking_ros_node', '--ros-args', '-r',
             '/camera/color/image_raw:=/x500/camera'],
        # cmd=['./tracking_ros_node'],
        cwd='/home/ernie/thesis/track/src/detection/build',
        output='screen'
    )

    estimation = ExecuteProcess(
        cmd=["/home/ernie/thesis/track/src/estimation/build/estimation_node",
             '--ros-args', '-r', '/camera/color/camera_info:=/x500/camera_info'],
        # cmd=["/home/ernie/thesis/track/src/estimation/build/estimation_node"],
        # prefix=['xterm -e gdb -ex run --args'],
        output='screen'
    )
    # estimation = ExecuteProcess(
    #     cmd=["/home/ernie/thesis/track/src/estimation/build/filter_exe"],
    #     prefix=['xterm -e gdb   -ex run --args'],
    #     output='screen'
    # )

    uncompress = ExecuteProcess(
        cmd=['ros2', 'run', 'image_transport', 'republish', 'compressed', 'raw', '--ros-args', '-r',
             '/in/compressed:=/camera/color/image_raw/compressed', '-r', 'out:=/camera/color/image_raw'],
        output='screen'
    )

    # play_bag_cmd = '''ros2 bag play latest-niceish-very-far/ --start-offset 80'''  #
    play_bag_cmd = '''ros2 bag play rosbag2_2023_10_20-14_06_24'''  #
    play_bag = ExecuteProcess(
        cmd=play_bag_cmd.split(),
        cwd="/home/ernie/thesis/ros_ws",
        # cwd="/home/ernie/thesis/bags",
        output='screen'
    )

    robot_state_pub = ExecuteProcess(
        cmd=['ros2', 'run', 'robot_state_publisher',
             'robot_state_publisher', 'cam.urdf'],
        cwd="/home/ernie/thesis/track",
        output='screen'
    )

    return LaunchDescription([
        play_bag,
        tracking,
        estimation,
        robot_state_pub,
        # uncompress,
        imu_mag_repub,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    dir_path,
                    "madgwick.launch.py"
                ])
            ]),
        ),
    ])
