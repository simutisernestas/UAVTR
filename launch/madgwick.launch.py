import os
import launch
import launch.actions
import launch.substitutions
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    config_dir = os.path.join(get_package_share_directory(
        'imu_filter_madgwick'), 'config')

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package='imu_filter_madgwick',
                executable='imu_filter_madgwick_node',
                name='imu_filter',
                output='screen',
                parameters=[
                    os.path.join(config_dir, 'imu_filter.yaml'),
                    {'use_mag': True},  # , 'mag_bias_x': 0.33534499,
                    # 'mag_bias_y': 0.02863634, 'mag_bias_z': 0.94166007
                ],
                remappings=[
                    ('imu/data_raw', '/x500/imu'),
                    ('imu/mag', '/x500/magnetometer'),
                ]
            )
        ]
    )
