from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
import os


def generate_launch_description():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # usbreset "Intel(R) RealSense(TM) Depth Camera 455"

    xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'serial', '--dev', '/dev/ttyUSB0', '-b', '921600'],
        output='screen'
    )

    sensor_tfs = ExecuteProcess(
        cmd=["python3", os.path.join(dir_path, "pub_tf.py")],
        output='screen'
    )

    # panda@panda:~/ros2_ws/scripts$ cat /etc/udev/rules.d/10-local.rules 
    # ACTION=="add", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", SYMLINK+="ttyTERA"
    altimeter_node = Node(
        package='teraranger_ros2',
        executable='teraranger_ros2_node',
        name='teraranger_ros2_node',
        output='screen',
        arguments=["/dev/ttyTERA"]
    )

    return LaunchDescription([
        xrce_agent,
        sensor_tfs,
        altimeter_node,
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([
        #         PathJoinSubstitution([
        #             dir_path,
        #             "rs_launch.py"
        #         ])
        #     ]),
        # )
    ])