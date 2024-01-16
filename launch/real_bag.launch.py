from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os

Q = [2.271574526742034e-05, -4.865236244499512e-05, 6.87666678825161e-05, 0.0003791923966091262, 0.0002985261641058709, 0.000319097897116446, 0.00023138896303178874, -7.07748832200966e-05, 0.00020829008799675924, -4.865236244499512e-05, 0.00010785352803227254, -0.00032831052084947396, -0.0008755336761758735, -0.0007853042546830924, -0.0009086342936896437, -0.00058999753673057, 6.530746135618034e-05, -0.0005651810440614356, 6.87666678825161e-05, -0.00032831052084947396, 0.009189060520762604, 0.0042386885668925615, 0.008166746613183607, 0.012137733624392015, 0.005384358184053825, 0.003987878531136719, 0.00654478713857567, 0.0003791923966091262, -0.0008755336761758735, 0.0042386885668925615, 0.008213630253438603, 0.00712364473122256, 0.009175489750994291, 0.005465601293933519, 0.0014877725466054589, 0.005392013675195147, 0.0002985261641058709, -0.0007853042546830924, 0.008166746613183607, 0.00712364473122256,
     0.009957030861520173, 0.013225454639567029, 0.006831632088542857, 0.0019526815252770505, 0.00756105157660486, 0.000319097897116446, -0.0009086342936896437, 0.012137733624392015, 0.009175489750994291, 0.013225454639567027, 0.018380318797488426, 0.009072754399557204, 0.004225327877307334, 0.01028684418235599, 0.00023138896303178874, -0.00058999753673057, 0.005384358184053825, 0.005465601293933518, 0.006831632088542857, 0.009072754399557204, 0.0066982359678246055, 0.0019381348390502945, 0.0054998476042949435, -7.07748832200966e-05, 6.530746135618034e-05, 0.003987878531136719, 0.0014877725466054589, 0.00195268152527705, 0.004225327877307334, 0.001938134839050295, 0.006266586347672024, 0.002217168575837674, 0.00020829008799675924, -0.0005651810440614356, 0.00654478713857567, 0.005392013675195147, 0.00756105157660486, 0.010286844182355988, 0.005499847604294944, 0.002217168575837674, 0.0070940910325205265]
pos_R = [4.148242398539099, 0.0, 0.0, 5.702412298355506]
vel_R = [0.16966872074959238, 0.0, 0.0, 0.0,
         0.06049256508421064, 0.0, 0.0, 0.0, 0.013327101067543388]
acc_R = [3.652282205366519, 0.0, 0.0, 0.0,
         4.333955375627488, 0.0, 0.0, 0.0, 0.23400083776894456]


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
