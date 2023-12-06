from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os


def generate_launch_description():
    root_dir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))) # parent; project root

    imu_mag_repub = ExecuteProcess(
        cmd=[f"{root_dir}/src/estimation/build/orientation_filter"],
        output='screen'
    )

    tracking = ExecuteProcess(
        cmd=['./tracking_ros_node', '--ros-args', '-r',
             '/camera/color/image_raw:=/x500/camera'],
        cwd=f'{root_dir}/src/detection/build',
        output='screen'
    )

    estimation = ExecuteProcess(
        cmd=[f"{root_dir}/src/estimation/build/estimation_node",
             '--ros-args', '-r', '/camera/color/camera_info:=/x500/camera_info',
             '-r', '/camera/color/image_raw:=/x500/camera'],
        # prefix=['xterm -fa "Monospace" -fs 14 -e gdb -tui -iex break -ex "b main" -ex run --args'],
        output='screen'
    )

    raise NotImplementedError("TODO: no bag")
    play_bag_cmd = '''ros2 bag play rosbag2_2023_10_20-14_06_24'''  #
    play_bag = ExecuteProcess(
        cmd=play_bag_cmd.split(),
        cwd=f"{root_dir}/bags",
        output='screen'
    )

    robot_state_pub = ExecuteProcess(
        cmd=['ros2', 'run', 'robot_state_publisher',
             'robot_state_publisher', 'cam.urdf'],
        cwd=f"{root_dir}/assets",
        output='screen'
    )

    return LaunchDescription([
        play_bag,
        tracking,
        estimation,
        robot_state_pub,
        imu_mag_repub,
    ])
