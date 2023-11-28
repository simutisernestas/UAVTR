# %%
import numpy as np
import pandas as pd

# %%

df = pd.read_csv("pjdata.csv")


# %%

# time:440.524
# prev_time:440.257
# cam_R_enu: -0.147838  -0.786657   0.599429
#  -0.988639   0.134176 -0.0677446
#  -0.027137  -0.602634  -0.797556
# height:8.80421
# r: 0.0941378
# -0.0717118
# -0.0879915

# //      4) PJ averaged angular velocity from camera gyro
# //      7) PJ ground truth velocity
# //      8) PJ averaged angular velocity of the drone

t0 = 440.257
t1 = 440.524

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
gt_velocity

# %%

# time_df = pd.read_csv("time.csv")
# time_offset_col = "/fmu/out/timesync_status/observed_offset"
# time_df[time_df[time_offset_col].notna()][time_offset_col].mean() / 1e6
time_off = -1697624417.8292115
time_off

# %%

ang_vel_cam_gyro = df[df["/camera/imu/header/stamp"].notna()]
ang_vel_cam_gyro["/camera/imu/header/stamp"] += time_off
ang_vel_cam_gyro = ang_vel_cam_gyro[(
    ang_vel_cam_gyro["/camera/imu/header/stamp"] > t0) & (ang_vel_cam_gyro["/camera/imu/header/stamp"] < t1)]
ang_vel_cam_gyro = ang_vel_cam_gyro[["/camera/imu/angular_velocity/x",
                                     "/camera/imu/angular_velocity/y",
                                     "/camera/imu/angular_velocity/z"]]
cam_ang_vel = ang_vel_cam_gyro.mean().to_numpy()
cam_ang_vel

# %%

# angular vel of the drone from /imu/data_raw topic
ang_vel_drone = df[df["/imu/data_raw/header/stamp"].notna()]
ang_vel_drone = ang_vel_drone[(
    ang_vel_drone["/imu/data_raw/header/stamp"] > t0) & (ang_vel_drone["/imu/data_raw/header/stamp"] < t1)]
ang_vel_drone = ang_vel_drone[["/imu/data_raw/angular_velocity/x",
                               "/imu/data_raw/angular_velocity/y",
                               "/imu/data_raw/angular_velocity/z"]]
drone_ang_vel = ang_vel_drone.mean().to_numpy()
drone_ang_vel


