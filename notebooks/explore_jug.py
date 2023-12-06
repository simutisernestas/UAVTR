# %%
from scipy.signal import butter, lfilter
import okf
import torch
from scipy.signal import savgol_filter
import scipy.optimize as opt
import pymap3d as pm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy import interpolate
np.set_printoptions(precision=6, suppress=True)


# %%

# read cvs file record.csv
df = pd.read_csv('../state_float.csv')
df.columns, len(df)

# %%
lat, lng, alt = 55.602979999999995, 12.3868665, 1.0210000000000001

# %%

# iterate all the rows
gt_pos = []
gt_time = []
meas_pos = []
meas_time = []
acc = []
acc_time = []
est_pos = []
est_time = []
vel_meas = []
vel_time = []
for i, row in df.iterrows():
    if not np.isnan(row["/gps_postproc/altitude"]):
        dlat = row["/gps_postproc/latitude"]
        dlng = row["/gps_postproc/longitude"]
        dalt = row["/gps_postproc/altitude"]
        time = row["/gps_postproc/header/stamp"]
        pos = pm.geodetic2enu(lat, lng, alt, dlat, dlng, dalt)
        gt_pos.append(pos)
        gt_time.append(time)

    if not np.isnan(row["/vec_target/points.1/x"]):
        pos = (row["/vec_target/points.1/x"],
               row["/vec_target/points.1/y"],
               row["/vec_target/points.1/z"])
        meas_pos.append(pos)
        meas_time.append(row["/vec_target/header/stamp"])

    if not np.isnan(row["/imu/data_raw/header/stamp"]):
        acc.append([
            row["/imu/data_raw/linear_acceleration/x"],
            row["/imu/data_raw/linear_acceleration/y"],
            row["/imu/data_raw/linear_acceleration/z"]
        ])
        acc_time.append(row["/imu/data_raw/header/stamp"])

    if not np.isnan(row["/state/header/stamp"]):
        pos = [row["/state/poses.0/position/x"],
               row["/state/poses.0/position/y"],
               row["/state/poses.0/position/z"]]
        est_pos.append(pos)
        est_time.append(row["/state/header/stamp"])

    if not np.isnan(row["/imu/data_world/header/stamp"]):
        vel_meas.append([
            row["/imu/data_world/vector/x"],
            row["/imu/data_world/vector/y"],
            row["/imu/data_world/vector/z"]
        ])
        vel_time.append(row["/imu/data_world/header/stamp"])

gt_pos = np.array(gt_pos)
meas_pos = np.array(meas_pos)
acc = np.array(acc)
acc_time = np.array(acc_time)
gt_time = np.array(gt_time)
meas_time = np.array(meas_time)
est_pos = np.array(est_pos)
est_time = np.array(est_time)
vel_meas = np.array(vel_meas)
vel_time = np.array(vel_time)
print(len(gt_pos), len(meas_pos), len(acc), len(est_pos), len(vel_meas))

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# %%

non_nan_idx = df["/imu/data_world/header/stamp"].notna()
stamp = df["/imu/data_world/header/stamp"][non_nan_idx]
xyz = df[["/imu/data_world/vector/x",
    "/imu/data_world/vector/y",
    "/imu/data_world/vector/z"]][non_nan_idx]
xyz.to_numpy().var(axis=0) # variance

# %%

plt.figure(dpi=200)
plt.scatter(gt_time, gt_pos[:, 0], label="X ground truth")
plt.scatter(est_time, est_pos[:, 0], label="X estimated")
plt.scatter(meas_time, meas_pos[:, 0], label="X measured")
plt.legend()
plt.figure(dpi=200)
plt.scatter(gt_time, gt_pos[:, 1], label="Y ground truth")
plt.scatter(est_time, est_pos[:, 1], label="Y estimated")
plt.scatter(meas_time, meas_pos[:, 1], label="Y measured")
plt.legend()
plt.figure(dpi=200)
plt.scatter(gt_time, gt_pos[:, 2], label="Z ground truth")
plt.scatter(est_time, est_pos[:, 2], label="Z estimated")
plt.scatter(meas_time, meas_pos[:, 2], label="Z measured")
plt.legend()
plt.figure(dpi=200)
plt.scatter(gt_time, np.linalg.norm(gt_pos, axis=1), label="ground truth norm")
plt.scatter(est_time, np.linalg.norm(est_pos, axis=1), label="estimated norm")
plt.legend()

# %%

# # filter_order = 2
# # cutoff_frequency = 1
# # b, a = butter(filter_order, cutoff_frequency,
# #                 fs=200, btype="lowpass", analog=False)
# # filtered_acc_x = lfilter(b, a, acc[:, 0])
# # filtered_acc_y = lfilter(b, a, acc[:, 1])
# # filtered_acc_z = lfilter(b, a, acc[:, 2])
# # acc = np.stack([filtered_acc_x, filtered_acc_y, filtered_acc_z], axis=1)
# # plt.plot(acc)
# # N = 2
# # AXIS = 2
# # plt.scatter(acc_time[:-(N-1)],
# #             moving_average(acc[:, AXIS], n=N), label="acc", s=.1)
# # plt.scatter(gt_time, gt_pos[:, AXIS] / 20, label="X pos gt", s=.1)
# # plt.legend()

# # %%

# x = [p3,v3,a3] 9 states in total
# y = [p3,v3,a3] 9 measurements in total
# A = [[0, 1, 0],
#      [0, 0, -1],
#      [0, 0, 0]]
N_STATES = 9
A = np.block([[np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
              [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)],
              [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],])
B = np.zeros((N_STATES, 3))
# C = [[1, 0, 0],
#      [0, 0, 1]]
C = np.block([[np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))],
              [np.zeros((3, 3)), -np.eye(3), np.zeros((3, 3))],
              [np.zeros((3, 3)), np.zeros((3, 3)), -np.eye(3)]])
# D = [[0], [0]]
D = np.zeros((9, 3))

sys = ct.ss(A, B, C, D)
obs = ct.obsv(sys.A, sys.C)
print("Obs:", np.linalg.matrix_rank(obs))

# discretize A
dt = 1/128
print("dt:", dt)
Ad = la.expm(A*dt)
np.set_printoptions(suppress=True, precision=7)
Ad, Ad.shape

# # print Ad in nice format no line breaks in single row
# for row in Ad:
#     print("[", end="")
#     for el in row:
#         print(f"{el:.7e},", end="")
#     print("],")

# # %%

# # Define the system matrices
# F = Ad

# H = np.array(C)

# # Q = np.eye(9)
# # Q[:3, :3] *= 5
# # Q[3:6, 3:6] *= 25
# # Q[6:, 6:] *= 2
# # R = np.eye(6)
# # R[:3, :3] *= 1
# # R[3:, 3:] *= 1
# Q = np.array([[1.4327232e-05,4.5542407e-05,-3.6177772e-05,3.8054527e-05,-3.7679753e-05,3.7981345e-05,-3.6543158e-05,6.6228309e-05,-1.5780837e-05,],
# [4.5542407e-05,1.5206209e-04,-8.7649881e-05,1.4784123e-04,-9.2957250e-05,1.4787818e-04,-1.6386951e-04,1.9778008e-04,-8.1902863e-05,],
# [-3.6177772e-05,-8.7649881e-05,1.9678981e-04,-1.2330969e-05,1.7859111e-04,2.2915255e-05,-7.7550530e-05,-1.9246018e-04,-6.0079266e-05,],
# [3.8054527e-05,1.4784123e-04,-1.2330969e-05,4.0092404e-04,4.2468562e-05,2.0720649e-04,1.9879875e-04,3.7377742e-04,-3.4079141e-04,],
# [-3.7679753e-05,-9.2957250e-05,1.7859111e-04,4.2468562e-05,3.8693461e-04,-2.2824766e-04,-5.0098784e-04,-1.2712694e-04,-2.5829339e-05,],
# [3.7981345e-05,1.4787818e-04,2.2915255e-05,2.0720649e-04,-2.2824766e-04,5.3843447e-04,3.5249679e-04,6.1598909e-05,-7.0989167e-05,],
# [-3.6543158e-05,-1.6386951e-04,-7.7550530e-05,1.9879875e-04,-5.0098784e-04,3.5249679e-04,1.5456325e-01,6.5116601e-02,1.1024973e-02,],
# [6.6228309e-05,1.9778008e-04,-1.9246018e-04,3.7377742e-04,-1.2712694e-04,6.1598909e-05,6.5116601e-02,1.2050196e-01,1.9366697e-02,],
# [-1.5780837e-05,-8.1902863e-05,-6.0079266e-05,-3.4079141e-04,-2.5829339e-05,-7.0989167e-05,1.1024973e-02,1.9366697e-02,6.7944942e-02,],
# ])
# R = np.array([[1.7650208e+01,1.2699096e+01,2.0122929e+00,1.3650661e+00,-1.4133519e+00,3.0783753e-01,-1.8889387e+00,-8.6473987e-01,-2.4666565e-01,],
# [1.2699096e+01,1.5283947e+01,2.0745422e+00,1.3172493e+00,-1.7365055e+00,-3.0467313e-01,-1.5104498e+00,-1.2713874e+00,-1.7304563e-01,],
# [2.0122929e+00,2.0745422e+00,2.3927872e+00,-1.0858562e-01,-3.7704457e-01,7.5956990e-02,-2.0219362e-01,-2.7574179e-01,-2.4521683e-01,],
# [1.3650661e+00,1.3172493e+00,-1.0858562e-01,3.3457488e+00,2.0952617e+00,5.4171535e-01,-5.2788083e-02,-6.1661927e-02,-1.4297597e-01,],
# [-1.4133519e+00,-1.7365055e+00,-3.7704457e-01,2.0952617e+00,5.3969640e+00,7.7712646e-01,2.0171614e-01,-2.4512484e-02,-7.9229035e-02,],
# [3.0783753e-01,-3.0467313e-01,7.5956990e-02,5.4171535e-01,7.7712646e-01,1.7101273e+00,1.1092571e-01,-5.8702137e-02,2.5037332e-02,],
# [-1.8889387e+00,-1.5104498e+00,-2.0219362e-01,-5.2788083e-02,2.0171614e-01,1.1092571e-01,4.3669211e+00,3.1101683e-01,-7.3201366e-02,],
# [-8.6473987e-01,-1.2713874e+00,-2.7574179e-01,-6.1661927e-02,-2.4512484e-02,-5.8702137e-02,3.1101683e-01,4.1195669e+00,-3.5575474e-01,],
# [-2.4666565e-01,-1.7304563e-01,-2.4521683e-01,-1.4297597e-01,-7.9229035e-02,2.5037332e-02,-7.3201366e-02,-3.5575474e-01,1.9104830e+00,],
# ])

# # Initial state estimate
# x_hat = np.zeros((9, 1))
# x_hat[:3] = meas_pos[0].reshape(3, 1)

# # Initial error covariance
# P = np.eye(9) * 10000

# # data prep
# # print(len(meas_pos), len(acc))
# first_acc_index = np.searchsorted(acc_time, meas_time[0])
# print(first_acc_index)
# last_pos_meas_index = 0
# timestamp = acc_time[first_acc_index-1]

# record = []
# # Kalman Filter loop
# for i in range(acc[first_acc_index:].shape[0]):
#     dt = acc_time[first_acc_index+i] - timestamp
#     timestamp = acc_time[first_acc_index+i]

#     # Predict step
#     x_hat = F @ x_hat
#     P = F @ P @ F.T + Q

#     p_meas = None
#     acc_meas_time = acc_time[first_acc_index+i]
#     if last_pos_meas_index == len(meas_time):
#         print(f"Break on: {i}")
#         break
#     if acc_meas_time > meas_time[last_pos_meas_index]:
#         p_meas = meas_pos[last_pos_meas_index].reshape(3, 1)
#         last_pos_meas_index += 1

#     # Position update
#     if p_meas is not None:
#         # p_meas = meas_pos[last_pos_meas_index].reshape(3, 1)
#         Hp = H[:3, :]
#         K = P @ Hp.T @ np.linalg.inv(Hp @ P @ Hp.T + R[:3, :3])
#         x_hat = x_hat + K @ (p_meas - Hp @ x_hat)
#         P = (np.eye(N_STATES) - K @ Hp) @ P

#     # Acceleration update
#     Ha = H[3:, :]
#     K = P @ Ha.T @ np.linalg.inv(Ha @ P @ Ha.T + R[3:, 3:])
#     x_hat = x_hat + K @ (acc[i].reshape(3, 1) - Ha @ x_hat)
#     P = (np.eye(N_STATES) - K @ Ha) @ P

#     record.append(x_hat.T)

# print(P)

# record = np.array(record).reshape(-1, 9)
# plt.figure(dpi=200)  # norm
# plt.plot(acc_time[first_acc_index:first_acc_index+len(record)], np.linalg.norm(
#     record[:, :3], axis=1), label="KF norm")
# plt.plot(gt_time, np.linalg.norm(gt_pos, axis=1), label="gt norm")
# plt.plot(meas_time, np.linalg.norm(meas_pos, axis=1),
#          label="meas norm", linestyle="--")
# # plt.ylim(-10,40)
# plt.legend()
# plt.show()

# # plot axes of estimation
# plt.figure(dpi=200)
# plt.plot(acc_time[first_acc_index:first_acc_index+len(record)],
#          record[:, 0], label="KF X")
# plt.plot(acc_time[first_acc_index:first_acc_index+len(record)],
#             record[:, 1], label="KF Y")
# plt.plot(acc_time[first_acc_index:first_acc_index+len(record)],
#             record[:, 2], label="KF Z")
# plt.legend()
# plt.show()


# %%

interpolated_meas_x = np.interp(acc_time, meas_time, meas_pos[:, 0])
interpolated_meas_y = np.interp(acc_time, meas_time, meas_pos[:, 1])
interpolated_meas_z = np.interp(acc_time, meas_time, meas_pos[:, 2])
interpolated_meas = np.stack(
    [interpolated_meas_x, interpolated_meas_y, interpolated_meas_z], axis=1)

interpolated_vel_meas_x = np.interp(acc_time, vel_time, vel_meas[:, 0])
interpolated_vel_meas_y = np.interp(acc_time, vel_time, vel_meas[:, 1])
interpolated_vel_meas_z = np.interp(acc_time, vel_time, vel_meas[:, 2])
interpolated_vel_meas = np.stack(
    [interpolated_vel_meas_x, interpolated_vel_meas_y, interpolated_vel_meas_z], axis=1)


def interpolate_gt_pos(t, pos):
    interp_F = interpolate.interp1d(
        t, pos, kind='cubic', fill_value="extrapolate")
    interpolated_pos = interp_F(acc_time)
    return interpolated_pos


def get_vel_and_acc(t, pos):
    interp_F = interpolate.interp1d(
        t, pos, kind='cubic', fill_value="extrapolate")
    interpolated_pos = interp_F(acc_time)
    # smooth position
    delta_time = np.diff(acc_time)
    delta_pose = np.diff(interpolated_pos)
    vel = delta_pose / delta_time
    # smooth velocity
    print(delta_pose.shape)
    vel = savgol_filter(vel, 200, 3)
    delta2_pose = np.diff(vel)
    acc = delta2_pose / delta_time[1:]
    return vel, acc


# interpolate gt position
interpolated_gt_x = interpolate_gt_pos(gt_time, gt_pos[:, 0])
interpolated_gt_y = interpolate_gt_pos(gt_time, gt_pos[:, 1])
interpolated_gt_z = interpolate_gt_pos(gt_time, gt_pos[:, 2])

print(gt_time.shape, gt_pos.shape)
gt_vel_x, gt_acc_x = get_vel_and_acc(gt_time, gt_pos[:, 0])
gt_vel_y, gt_acc_y = get_vel_and_acc(gt_time, gt_pos[:, 1])
gt_vel_z, gt_acc_z = get_vel_and_acc(gt_time, gt_pos[:, 2])

interpolated_gt_pos = np.stack([interpolated_gt_x, interpolated_gt_y,
                                interpolated_gt_z], axis=1)
gt_vel = np.stack([gt_vel_x, gt_vel_y, gt_vel_z], axis=1)
gt_acc = np.stack([gt_acc_x, gt_acc_y, gt_acc_z], axis=1)

plt.figure(dpi=300)
plt.scatter(acc_time, acc[:, 0], label="real", s=.1)
plt.scatter(acc_time[:-2], gt_acc_x, label="GT", s=.1)
plt.legend()
# plt.show()

# plt.plot(acc_time, interpolated_gt_pos, label="gt")
# plt.plot(acc_time, interpolated_meas, linestyle="--", label="meas")

# %%


def get_F():
    return torch.tensor(Ad, dtype=torch.double)


def get_H():
    return torch.tensor(C, dtype=torch.double)


def initial_observation_to_state(z):
    # z = [p,a]
    Ct = torch.tensor(C, dtype=torch.double)
    return Ct @ z


def loss_fun():
    return lambda pred, x: ((pred-x)**2).sum()


def model_args():
    return dict(
        dim_x=9,
        dim_z=9,
        init_z2x=initial_observation_to_state,
        F=get_F(),
        H=get_H(),
        loss_fun=loss_fun(),
    )


# %%

X = [np.hstack((interpolated_gt_pos[:-2, :],
               gt_vel[:-1, :], -gt_acc)).astype(np.float64)]
Z = [np.hstack((interpolated_meas[:-2],
               interpolated_vel_meas[:-2], acc[:-2])).astype(np.float64)]

X[0].shape, Z[0].shape

# %%

# Define model
okf_model_args = model_args()
print('---------------\nModel arguments:\n', okf_model_args)
model = okf.OKF(**okf_model_args, optimize=True, model_name='OKF_REAL')

# %%


def print_for_copy(np_array, matrix):
    print(f"{matrix} = np.array([", end="")
    for row in np_array:
        print("[", end="")
        for el in row:
            print(f"{el:.7e},", end="")
        print("],")
    print("])")


model.estimate_noise(X, Z)
print_for_copy(model.get_Q(), "Q")
print_for_copy(model.get_R(), "R")

# %%

# Run training

print_for_copy(model.get_Q(), 'Q')
print_for_copy(model.get_R(), 'R')

# model.estimate_noise(X,Z)

res, _ = okf.train(model, Z, X, verbose=1, n_epochs=3,
                   batch_size=1, to_save=False)

print_for_copy(model.get_Q(), 'Q')
print_for_copy(model.get_R(), 'R')
# %%
