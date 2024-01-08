from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os

Q = [1.074029971005409e-05, -6.990778107075084e-06, 2.8879696542461248e-05, -3.896123610237218e-05, -5.026431695538006e-06, 2.379373213949113e-05, 0.00015039224301388195, -1.3905482867629694e-05, 7.389555529783657e-05, -6.990778107075084e-06, 9.875338958994674e-06, 0.0001367605489803695, 6.559750744530176e-05, 0.00012153376724882262, 0.00013142342648145544, -0.00013911515901774167, -2.2647752237425523e-05, -7.316820987015946e-05, 2.8879696542461248e-05, 0.0001367605489803695, 0.004625735496464513, 0.0012717059679190996, 0.0034187838554920152, 0.004515625044062059, -0.0007376360636122373, -0.0008870645290879898, -0.0005014355782078776, -3.896123610237218e-05, 6.559750744530176e-05, 0.0012717059679190996, 0.010876975055954122, -0.0002649328165915387, 0.009473585552952869, 0.0023196104910681277, 0.0038045360681695814, 0.00120822465836315, -5.026431695538006e-06, 0.00012153376724882262, 0.0034187838554920152, -0.0002649328165915387,
     0.002840102639056131, 0.0014181606806617738, -0.0010629986233079082, -0.001307476444972363, -0.0007565091412223148, 2.379373213949113e-05, 0.00013142342648145544, 0.004515625044062059, 0.009473585552952869, 0.001418160680661774, 0.023480668334735066, -0.000590041885712129, 0.0032476803969755438, 0.0009769215578316988, 0.00015039224301388195, -0.00013911515901774167, -0.0007376360636122373, 0.0023196104910681277, -0.0010629986233079082, -0.000590041885712129, 0.007936435290093444, -0.0010233603605527915, 0.0017254547358859728, -1.3905482867629694e-05, -2.2647752237425523e-05, -0.0008870645290879898, 0.0038045360681695814, -0.001307476444972363, 0.0032476803969755438, -0.0010233603605527917, 0.0065189429468531495, 0.001001264995538991, 7.389555529783657e-05, -7.316820987015946e-05, -0.0005014355782078776, 0.0012082246583631497, -0.0007565091412223148, 0.000976921557831699, 0.0017254547358859728, 0.001001264995538991, 0.003646661063990259]
pos_R = [2.5312946013877826, -2.7717861238525168, -
         2.7717861238525168, 5.929743177599134]
vel_R = [0.22956343628856626, -0.11534638549474974, -0.01257244095560051, -0.11534638549474974,
         0.20037558709757752, 0.04302822082224472, -0.01257244095560051, 0.043028220822244716, 0.13796190145459955]
acc_R = [3.3575427759873047, -0.03751565219571708, 0.013977018844111847, -0.03751565219571709,
         3.964373046266632, -0.7157101439635045, 0.013977018844111847, -0.7157101439635046, 2.024693092361415]


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
