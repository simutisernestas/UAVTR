from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
    xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'serial', '--dev', '/dev/ttyUSB0', '-b', '921600'],
        output='screen'
    )

    rec_cmd = '''
        ros2 bag record --compression-mode message --compression-format zstd
            /fmu/out/sensor_baro
            /fmu/out/sensor_combined
            /fmu/out/sensor_mag
            /fmu/out/vehicle_attitude
            /fmu/out/vehicle_global_position
            /fmu/out/vehicle_local_position
            /fmu/out/vehicle_odometry
            /fmu/out/vehicle_gps_position
            /fmu/out/vehicle_magnetometer
            /fmu/out/timesync_status
    '''
    record_process = ExecuteProcess(
        cmd=rec_cmd.split(),
        output='screen'
    )

    return LaunchDescription([
        xrce_agent,
        TimerAction(period=5.0, actions=[record_process]),
    ])