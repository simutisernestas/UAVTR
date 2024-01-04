from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os

CAM_URDF = """
<?xml version="1.0"?>
<robot name="camera_robot">

  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2" />
      </geometry>
    </visual>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link" />
    <child link="camera_link" />
    <origin xyz="0 0 .2" rpy="0 0.3490658503988659 0" />
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.010 0.03 0.03" />
      </geometry>
    </visual>
  </link>

  <joint name="camera_optical_joint" type="fixed">
    <origin xyz="0 0 0" rpy="-1.5707963267948966 0 -1.5707963267948966" />
    <parent link="camera_link" />
    <child link="camera_link_optical" />
  </joint>

  <link name="camera_link_optical"></link>

</robot>
"""


def generate_launch_description():
    root_dir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))  # parent; project root

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

    raise NotImplementedError("no bag")
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
        output='screen',
        # NOT TESTED !!!
        parameters=[{'robot_description': CAM_URDF}]
    )

    return LaunchDescription([
        play_bag,
        tracking,
        estimation,
        robot_state_pub,
        imu_mag_repub,
    ])
