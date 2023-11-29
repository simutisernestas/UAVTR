# %%

import numpy as np
import pandas as pd

# %%

df = pd.read_csv("pjdata.csv")

file = open("flowinfo.txt", "r")
lines = file.readlines()
for line in lines:
    if line.startswith("time:"):
        time = float(line.split(":")[1])
    elif line.startswith("prev_time:"):
        prev_time = float(line.split(":")[1])
    elif line.startswith("cam_R_enu:"):
        scnd_line = lines[lines.index(line) + 1]
        third_line = lines[lines.index(line) + 2]
        cam_R_enu = np.array([float(x) for x in line.split(":")[1].split()] +
                             [float(x) for x in scnd_line.split()] +
                             [float(x) for x in third_line.split()])
        cam_R_enu = cam_R_enu.reshape((3, 3))
    elif line.startswith("height:"):
        height = float(line.split(":")[1])
    elif line.startswith("r:"):
        r0 = float(line.split(":")[1])
        r1 = float(lines[lines.index(line) + 1])
        r2 = float(lines[lines.index(line) + 2])
        r = np.array([r0, r1, r2])

t0 = prev_time - 1e-3 * 200
t1 = time + 1e-3 * 200

# df["/fmu/out/vehicle_gps_position/timestamp"]
# take where this is not none
gps_vel = df[df["/fmu/out/vehicle_gps_position/timestamp"].notna()]
gps_vel["/fmu/out/vehicle_gps_position/timestamp"] = gps_vel["/fmu/out/vehicle_gps_position/timestamp"] / 1e6
ts_columns = gps_vel["/fmu/out/vehicle_gps_position/timestamp"]
# filter out between t0 and t1
gps_vel = gps_vel[(ts_columns > t0) & (ts_columns < t1)]
gps_vel = gps_vel[["/fmu/out/vehicle_gps_position/vel_n_m_s",
                   "/fmu/out/vehicle_gps_position/vel_e_m_s",
                   "/fmu/out/vehicle_gps_position/vel_d_m_s"]]
gt_velocity = gps_vel.mean().to_numpy()
print(gt_velocity)

# %%

# time_df = pd.read_csv("time.csv")
# time_offset_col = "/fmu/out/timesync_status/observed_offset"
# time_df[time_df[time_offset_col].notna()][time_offset_col].mean() / 1e6
time_off = -1697624417.8292115
print(time_off)

# %%

ang_vel_cam_gyro = df[df["/camera/imu/header/stamp"].notna()]
ang_vel_cam_gyro["/camera/imu/header/stamp"] += time_off
ang_vel_cam_gyro = ang_vel_cam_gyro[(
                                            ang_vel_cam_gyro["/camera/imu/header/stamp"] > t0) & (
                                            ang_vel_cam_gyro["/camera/imu/header/stamp"] < t1)]
ang_vel_cam_gyro = ang_vel_cam_gyro[["/camera/imu/angular_velocity/x",
                                     "/camera/imu/angular_velocity/y",
                                     "/camera/imu/angular_velocity/z"]]
cam_ang_vel = ang_vel_cam_gyro.mean().to_numpy()
print(cam_ang_vel)

# %%

# angular vel of the drone from /imu/data_raw topic
ang_vel_drone = df[df["/imu/data_raw/header/stamp"].notna()]
ang_vel_drone = ang_vel_drone[(
                                      ang_vel_drone["/imu/data_raw/header/stamp"] > t0) & (
                                      ang_vel_drone["/imu/data_raw/header/stamp"] < t1)]
ang_vel_drone = ang_vel_drone[["/imu/data_raw/angular_velocity/x",
                               "/imu/data_raw/angular_velocity/y",
                               "/imu/data_raw/angular_velocity/z"]]
drone_ang_vel = ang_vel_drone.mean().to_numpy()
print(drone_ang_vel)

cpp_string = f"""
double t0 = {prev_time};
double t1 = {time};
Eigen::Matrix3d cam_R_enu;
cam_R_enu << {cam_R_enu[0][0]}, {cam_R_enu[0][1]}, {cam_R_enu[0][2]},
{cam_R_enu[1][0]}, {cam_R_enu[1][1]}, {cam_R_enu[1][2]},
{cam_R_enu[2][0]}, {cam_R_enu[2][1]}, {cam_R_enu[2][2]};
double height = {height};
Eigen::Vector3d r;
r << {r[0]}, {r[1]}, {r[2]};
Eigen::Vector3d cam_omega;
cam_omega << {cam_ang_vel[0]}, {cam_ang_vel[1]}, {cam_ang_vel[2]};
Eigen::Vector3d drone_omega;
drone_omega << {drone_ang_vel[0]}, {drone_ang_vel[1]}, {drone_ang_vel[2]};
Eigen::Vector3d gt_vel_ned = {{{gt_velocity[0]}, {gt_velocity[1]}, {gt_velocity[2]}}};
"""
print(cpp_string)
