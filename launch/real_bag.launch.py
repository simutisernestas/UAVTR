from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os

Q = [1.2943884954007733e-05, -5.85152455372156e-05, -1.2933872351222466e-05, 0.00036027120453046673, 0.00017742321513557134, -0.00025493066408782694, 5.1813297993141705e-05, 0.00011749569108770014, 3.0224745866284836e-05, -5.85152455372156e-05, 0.0002677913541318857, 6.443053010675914e-05, -0.0017343415668386224, -0.0008681871764740029, 0.0014075401536719336, -0.00026177760789416343, -0.0004581430047222851, -0.00011343504217358193, -1.2933872351222466e-05, 6.443053010675914e-05, 2.6551097848589227e-05, -0.0005884111107991327, -0.0003094051982195111, 0.0008422158491048875, -0.00012894637584900845, 2.8806104873371186e-05, 1.9246651898310607e-05, 0.00036027120453046673, -0.0017343415668386224, -0.0005884111107991327, 0.013909418530264289, 0.00722961554021448, -0.016929544817825902, 0.0026786297868602055, 0.0007345868184209928, -1.9789200697438767e-06, 0.00017742321513557134, -0.0008681871764740029, -0.0003094051982195111, 0.007229615540214479,
     0.0038243043775202943, -0.009170707266771896, 0.0013755981736671657, 6.910616712630714e-05, -8.643746316212364e-05, -0.00025493066408782694, 0.0014075401536719336, 0.0008422158491048874, -0.016929544817825902, -0.009170707266771898, 0.030357554200719103, -0.004362294941847995, 0.003969681179110004, 0.0015326957033691057, 5.1813297993141705e-05, -0.00026177760789416343, -0.00012894637584900845, 0.002678629786860206, 0.0013755981736671657, -0.004362294941847995, 0.0011993955022823527, -0.00038971589031127165, -0.00016876864341815816, 0.00011749569108770014, -0.0004581430047222851, 2.8806104873371186e-05, 0.0007345868184209928, 6.910616712630723e-05, 0.003969681179110004, -0.00038971589031127176, 0.0033023294029489673, 0.0004679464283494654, 3.0224745866284836e-05, -0.00011343504217358193, 1.9246651898310607e-05, -1.9789200697439445e-06, -8.643746316212364e-05, 0.0015326957033691057, -0.00016876864341815816, 0.0004679464283494654, 0.0011061273529560326]
pos_R = [7.611182408765501, -4.914267612196107, -
         4.914267612196107, 10.692050597531278]
vel_R = [0.21619070862570727, -0.025637267586592623, 0.12998039071691714, -0.025637267586592623,
         0.1628202269284226, 0.052160607485190816, 0.12998039071691714, 0.052160607485190816, 0.11228522188294238]
acc_R = [1.4197529150696722, 0.6616931980578991, 0.07106021130372175, 0.6616931980578991,
         2.4786680648519095, 0.2495506920146051, 0.07106021130372187, 0.24955069201460534, 2.572721167646211]

def launch_setup(context, *args, **kwargs):
    root_dir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))

    gain = 0.7
    magnetic_rejection = 0.0
    acceleration_rejection = 15.0
    recovery_trigger_period = 1
    params = ["--ros-args",
              "-p", f"gain:={gain}",
              "-p", f"magnetic_rejection:={magnetic_rejection}",
              "-p", f"acceleration_rejection:={acceleration_rejection}",
              "-p", f"recovery_trigger_period:={recovery_trigger_period}"]
    orientation_filter = ExecuteProcess(
        cmd=[f'{root_dir}/src/estimation/build/orientation_filter'] + params,
        output='screen',
    )

    tracking = ExecuteProcess(
        cmd=['./tracking_ros_node'],
        cwd=f'{root_dir}/src/detection/build',
        # prefix=['xterm  -e gdb -ex "run" --args'],
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
    BAG0_OFF = 130
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
        cwd=f"{root_dir}/scripts",
        output='screen'
    )

    baro_ref = 0.0
    if WHICH == 0:
        baro_ref = 25.94229507446289
    else:
        baro_ref = 7.0
    flow_err_threshold = 100.0
    estimation = ExecuteProcess(
        cmd=["./estimation_node", "--ros-args",
             "-p", f"baro_ground_ref:={baro_ref}",
             "-p", f"spatial_vel_flow_error:={flow_err_threshold}",
             "-p", f"flow_vel_rejection_perc:={10.0}",
             "-p", f"process_covariance:={Q}",
             "-p", f"pos_R:={pos_R}",
             "-p", f"vel_R:={vel_R}",
             "-p", f"acc_R:={acc_R}"],
        cwd=f'{root_dir}/src/estimation/build',
        # prefix=['xterm  -e gdb -ex "run" --args'],
        output='screen'
    )
    # wrap into timer to launch 2 seconds after everything
    est_timer = TimerAction(
        period=1.0,
        actions=[estimation]
    )

    return [
        play_bag,
        tracking,
        est_timer,
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
