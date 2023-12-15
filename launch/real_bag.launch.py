from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os


def launch_setup(context, *args, **kwargs):
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

    uncompress = ExecuteProcess(
        cmd=['ros2', 'run', 'image_transport', 'republish', 'compressed', 'raw', '--ros-args', '-r',
             '/in/compressed:=/camera/color/image_raw/compressed', '-r', 'out:=/camera/color/image_raw'],
        output='screen'
    )

    WHICH = int(context.launch_configurations['which'])
    MODE = int(context.launch_configurations['mode'])
    bag_name = ""
    offset = -1
    BAG0_OFF = 150
    if WHICH == 0:
        bag_name = "./18_0/rosbag2_2023_10_18-12_24_19"
        offset = BAG0_OFF
    else:
        bag_name = "./latest_flight/rosbag2_2023_10_18-16_22_16"
        modes = [
            1500,  # going to the moon
            1942,
            2275
        ]
        offset = modes[MODE]

    play_bag_cmd = f'''ros2 bag play {bag_name} --start-offset {offset}'''
    play_bag = ExecuteProcess(
        cmd=play_bag_cmd.split(),
        cwd=f"{root_dir}/bags",
        output='screen'
    )

    run_name = bag_name.split('/')[-2]
    if offset != BAG0_OFF:
        run_name += f'_mode{MODE}'

    record_state = ExecuteProcess(
        cmd=["python3", "record_state.py", run_name],
        cwd=f"{root_dir}/notebooks",
        output='screen'
    )

    baro_ref = 0.0
    if WHICH == 0:
        baro_ref = 25.94229507446289
    else:
        baro_ref = 7.0
    estimation = ExecuteProcess(
        cmd=["./estimation_node", "--ros-args",
             "-p", f"baro_ground_ref:={baro_ref}"],
        cwd=f'{root_dir}/src/estimation/build',
        # prefix=['xterm  -e gdb -ex "b main" --args'],
        output='screen'
    )

    return [
        play_bag,
        tracking,
        estimation,
        uncompress,
        orientation_filter,
        record_state
    ]


def generate_launch_description():
    # Declare the command line arguments
    which_arg = DeclareLaunchArgument(
        'which',
        default_value='0',  # 0 or 1
        description='Which argument'
    )

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='0',  # 0, 1, 2
        description='Mode argument'
    )

    evaluate_args = OpaqueFunction(function=launch_setup)

    return LaunchDescription([
        which_arg,
        mode_arg,
        evaluate_args
    ])
