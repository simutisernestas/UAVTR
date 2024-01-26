import numpy as np
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os

Q = [3.0077353579156e-05, 7.328882705298387e-05, 0.00011393196962562863, 6.18153192862859e-05, 0.0006458673789014268, 0.0003302590763730704, 5.253704653855038e-06, 0.00023673686903491839, -0.00047134470936069893, 7.328882705298387e-05, 0.00018485429962324516, 0.000286943709714482, 0.00021711684396088262, 0.001902514803154084, 0.0008095011981397754, -1.6465587905150844e-05, 0.0006376136606405004, -0.0010955534581037255, 0.00011393196962562863, 0.000286943709714482, 0.00045666458842511827, 0.0002550917217179434, 0.003344715591019234, 0.0012490665291629327, 8.799234432172386e-06, 0.0010548806853092794, -0.001598736608529918, 6.18153192862859e-05, 0.00021711684396088262, 0.00025509172171794337, 0.0013757050896509087, 0.0019738746718325055, 0.0007909428146400768, -0.0005277695597919655, 0.0006572610135035612, -0.0011644208713881316, 0.0006458673789014268, 0.001902514803154084, 0.003344715591019234, 0.001973874671832505,
     0.04603774733183946, 0.007008019804970036, -0.0002496127037407243, 0.01072607036797308, -0.003417954341041979, 0.0003302590763730704, 0.0008095011981397754, 0.0012490665291629327, 0.0007909428146400768, 0.007008019804970037, 0.0036408624474332336, 2.0864510516474854e-05, 0.0026013040250038562, -0.005215569812276277, 5.253704653855038e-06, -1.6465587905150844e-05, 8.799234432172386e-06, -0.0005277695597919656, -0.0002496127037407243, 2.0864510516474854e-05, 0.002476752431756734, -0.0006085756127973295, 0.0005761895186826065, 0.00023673686903491839, 0.0006376136606405004, 0.0010548806853092794, 0.0006572610135035612, 0.01072607036797308, 0.0026013040250038562, -0.0006085756127973294, 0.00502372669536296, -0.002560860456560353, -0.00047134470936069893, -0.0010955534581037255, -0.001598736608529918, -0.0011644208713881316, -0.003417954341041979, -0.005215569812276279, 0.0005761895186826065, -0.0025608604565603536, 0.01078417359102141]


Q = np.array(Q).reshape((9, 9))
STATES = 14
Qn = np.zeros((STATES, STATES))
Qn[:3, :3] = Q[:3, :3]
print(Qn[:3, :3])
Qn[3:6, 3:6] = Q[3:6, 3:6]
print(Qn[3:6, 3:6])
Qn[8:11, 8:11] = Q[6:9, 6:9]
print(Qn[8:11, 8:11])
dt = 1/128.0
Qn[6, 6] = 5*dt*1e-5
Qn[7, 7] = 5*dt*1e-5
print(Qn[6:8, 6:8])
Qn[11:, 11:] = np.eye(3)*5*dt*1e-3
print(Qn[11:, 11:])
Q = Qn.reshape(-1).tolist()
print(len(Q))

pos_R = [9.21526463438979, -2.129640717505567, -
         2.129640717505567, 6.9161431877491815]
vel_R = [0.7974188128024176, -0.11864072579366468, 0.3127028025737731, -0.11864072579366469,
         0.05644560687568999, 0.003255789290718028, 0.3127028025737731, 0.003255789290718028, 0.2379884575599689]
acc_R = [2.6644815080937105, -0.12254743545555724, -0.06631899114831313, -0.12254743545555724,
         5.927738509969938, -0.25012248592169417, -0.06631899114831313, -0.25012248592169417, 1.3102479434439718]


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
            1900,  # 1942,
            2245
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
    flow_err_threshold = 5.0
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
