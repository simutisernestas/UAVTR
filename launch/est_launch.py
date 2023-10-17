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

    # tracking = ExecuteProcess(
    #     cmd=['./tracking_ros_node'],
    #     cwd='/home/ernie/thesis/track/src/detection/build',
    #     output='screen'
    # )

    # estimation = ExecuteProcess(
    #     cmd=["/home/ernie/thesis/track/src/estimation/build/estimation_node"],
    #     output='screen'
    # )
    estimation = ExecuteProcess(
        cmd=["/home/ernie/thesis/track/src/estimation/build/filter_exe"],
        prefix=['xterm -e gdb   -ex run --args'],
        output='screen'
    )

    play_bag_cmd = '''ros2 bag play drone_data_asta/ --start-offset 60'''  # 
    play_bag = ExecuteProcess(
        cmd=play_bag_cmd.split(),
        cwd="/home/ernie/thesis/bags",
        output='screen'
    )

    return LaunchDescription([
        play_bag,
        # tracking,
        estimation,
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
