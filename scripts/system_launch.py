from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
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

    # record.sh
    # ros2 bag record --compression-mode message --compression-format zstd \
    #     /camera/accel/sample \
    #     /camera/color/camera_info \
    #     /camera/color/image_raw \
    #     /camera/gyro/sample \
    #     /camera/imu \
    #     /fmu/out/sensor_baro \
    #     /fmu/out/sensor_combined \
    #     /fmu/out/sensor_mag \
    #     /fmu/out/vehicle_attitude \
    #     /fmu/out/vehicle_global_position \
    #     /fmu/out/vehicle_local_position \
    #     /fmu/out/vehicle_odometry \
    #     /fmu/out/vehicle_gps_position \
    #     /fmu/out/vehicle_magnetometer \
    #     /tf_static \
    #     /teraranger_evo_40m \
    #     /fmu/out/timesync_status

    rec_cmd = '''
        ros2 bag record --compression-mode message --compression-format zstd
            /camera/accel/sample
            /camera/color/camera_info
            /camera/color/image_raw
            /camera/gyro/sample
            /camera/imu
            /fmu/out/sensor_baro
            /fmu/out/sensor_combined
            /fmu/out/sensor_mag
            /fmu/out/vehicle_attitude
            /fmu/out/vehicle_global_position
            /fmu/out/vehicle_local_position
            /fmu/out/vehicle_odometry
            /fmu/out/vehicle_gps_position
            /fmu/out/vehicle_magnetometer
            /fmu/out/vehicle_air_data
            /tf_static
            /teraranger_evo_40m
            /fmu/out/timesync_status
    '''
    record_process = ExecuteProcess(
        cmd=rec_cmd.split(),
        output='screen'
    )

    return LaunchDescription([
        xrce_agent,
        sensor_tfs,
        altimeter_node,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    dir_path,
                    "rs_launch.py"
                ])
            ]),
        ),
        TimerAction(period=10.0, actions=[record_process]),
    ])