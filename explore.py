# %%
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

# %%

# read cvs file record.csv
df = pd.read_csv('full1.csv')
df.columns, len(df)

# %%

# take the last row
# row = df.iloc[-1]
# remove this row
# df = df.iloc[:-1]

# lat, lng, alt = row[0], row[1], row[2]
lat, lng, alt = 55.602979999999995, 12.3868665, 1.0210000000000001

# %%

# iterate all the rows
gt_pos = []
gt_time = []
meas_pos = []
meas_time = []
acc = []
acc_time = []
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

# interpolate gt_pos to match meas_pos
gt_pos = np.array(gt_pos)
meas_pos = np.array(meas_pos)
acc = np.array(acc)
print(len(gt_pos), len(meas_pos), len(acc))

# remove all meas pos before acceleration
meas_pos = meas_pos[np.searchsorted(meas_time, acc_time[0]):]
meas_time = meas_time[np.searchsorted(meas_time, acc_time[0]):]


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# %%


plt.scatter(gt_time, gt_pos[:, 0], label="X gt")
plt.scatter(meas_time, meas_pos[:, 0], label="X meas")
plt.legend()
plt.figure()
plt.scatter(gt_time, gt_pos[:, 1], label="Y gt")
plt.scatter(meas_time, meas_pos[:, 1], label="Y meas")
plt.legend()
plt.figure()
plt.scatter(gt_time, gt_pos[:, 2], label="Z gt")
plt.scatter(meas_time, meas_pos[:, 2], label="Z meas")
plt.legend()
plt.show()
# plot norms
plt.scatter(gt_time, np.linalg.norm(gt_pos, axis=1), label="gt")
plt.scatter(meas_time, np.linalg.norm(meas_pos, axis=1), label="meas")
plt.legend()
plt.figure()
# N = 100
# plt.plot(acc_time[:-(N-1)], moving_average(acc[:, 0], n=N), label="acc")
# plt.legend()

# positive acc on y turns into negative
# negative acc on x turns into positive
# negative acc on z turns into positive
# seems to hold that it's just minus sign
# test it on !!! should have no significant drift over short horizon at least

# %%

# x = [p3,v3,a3] 9 states in total
# u = []
# y = [p3,a3] 6 measurements in total

# A = [[0, 1, 0],
#      [0, 0, 1],
#      [0, 0, 0]]
N_STATES = 9
A = np.block([[np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
              [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)],
              [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]])
B = np.zeros((N_STATES, 3))
# C = [[1, 0, 0],
#      [0, 0, 1]]
C = np.block([[np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))],
              [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)]])
# D = [[0], [0]]
D = np.zeros((6, 3))

sys = ct.ss(A, B, C, D)
obs = ct.obsv(sys.A, sys.C)
print("Obs:", np.linalg.matrix_rank(obs))

comb = np.concatenate([A, B], axis=1)
comb = np.concatenate([comb, np.zeros((3, 12))], axis=0)

# discretize
dt = 1/128
print("dt:", dt)
la.expm(comb*dt)
Ad = la.expm(comb*dt)[:N_STATES, :N_STATES]
Bd = la.expm(comb*dt)[:N_STATES, N_STATES:]
np.set_printoptions(precision=3, suppress=True)
Ad

# %%

# Define the system matrices
F = Ad

H = np.array(C)

Q = np.array([[0.0000099, 0.0000259, 0.0000035, -0.0001448, 0.0001073, -0.0000672, -0.0000053, 0.0000379, 0.0000362,],
              [0.0000259, 0.0000730, -0.0000900, -0.0004868, 0.0003638, -
                  0.0001648, 0.0000329, 0.0001457, 0.0000582,],
              [0.0000035, -0.0000900, 0.0019655, 0.0021422, -0.0016526, -
                  0.0002413, -0.0009383, -0.0009266, 0.0007556,],
              [-0.0001448, -0.0004868, 0.0021422, 0.0067042, -0.0052060,
               0.0011129, -0.0007343, -0.0020924, 0.0006546,],
              [0.0001073, 0.0003638, -0.0016526, -0.0052060, 0.0049054, -
               0.0016164, -0.0059811, -0.0011168, 0.0009459,],
              [-0.0000672, -0.0001648, -0.0002413, 0.0011129, -0.0016164,
               0.0017572, 0.0048352, -0.0017153, -0.0022121,],
              [-0.0000053, 0.0000329, -0.0009383, -0.0007343, -0.0059811,
               0.0048352, 0.1210067, 0.0544637, -0.0168264,],
              [0.0000379, 0.0001457, -0.0009266, -0.0020924, -
               0.0011168, -0.0017153, 0.0544637, 0.0965022, 0.0035058,],
              [0.0000362, 0.0000582, 0.0007556, 0.0006546, 0.0009459, -
               0.0022121, -0.0168264, 0.0035058, 0.0970379,],
              ])
R = np.array([[97.0645641, -54.4778585, -6.7935329, -0.8778010, -0.9570216, 0.5827924,],
              [-54.4778585, 47.5902880, 4.9473400, -
                  0.4048356, -1.2202665, -0.4457218,],
              [-6.7935329, 4.9473400, 1.0613884, 0.1964553, 0.0598765, -0.0421372,],
              [-0.8778010, -0.4048356, 0.1964553,
                  4.1640568, 0.5697341, 0.5803749,],
              [-0.9570216, -1.2202665, 0.0598765,
                  0.5697341, 3.0669205, 0.2549653,],
              [0.5827924, -0.4457218, -0.0421372,
                  0.5803749, 0.2549653, 1.5452825,],
              ])


# Initial state estimate
x_hat = np.zeros((9, 1))
x_hat[:3] = meas_pos[0].reshape(3, 1)

# Initial error covariance
P = np.eye(9) * 100

# data prep
print(len(meas_pos), len(acc))
timestamp = acc_time[0]
first_acc_index = np.searchsorted(acc_time, meas_time[0])
last_pos_meas_index = 0

record = []
# Kalman Filter loop
for i in range(acc[first_acc_index:].shape[0]):
    # Predict step
    x_hat = F @ x_hat
    P = F @ P @ F.T + Q

    p_meas = None
    acc_meas_time = acc_time[i]
    if last_pos_meas_index == len(meas_time):
        print(f"Break on: {i}")
        break
    if acc_meas_time > meas_time[last_pos_meas_index]:
        p_meas = meas_pos[last_pos_meas_index].reshape(3, 1)
        last_pos_meas_index += 1

    # Position update
    if p_meas is not None:
        Hp = H[:3, :]
        K = P @ Hp.T @ np.linalg.inv(Hp @ P @ Hp.T + R[:3, :3])
        x_hat = x_hat + K @ (p_meas - Hp @ x_hat)
        P = (np.eye(9) - K @ Hp) @ P

    # Acceleration update
    Ha = H[3:, :]
    K = P @ Ha.T @ np.linalg.inv(Ha @ P @ Ha.T + R[3:, 3:])
    x_hat = x_hat + K @ (-acc[i].reshape(3, 1) - Ha @ x_hat)
    P = (np.eye(N_STATES) - K @ Ha) @ P

    record.append(x_hat.T)

record = np.array(record).reshape(-1, 9)
plt.figure(dpi=200)  # norm
plt.plot(acc_time[first_acc_index:first_acc_index+len(record)], np.linalg.norm(
    record[:, :3], axis=1), label="KF norm")
plt.plot(gt_time, np.linalg.norm(gt_pos, axis=1), label="gt norm")
plt.plot(meas_time, np.linalg.norm(meas_pos, axis=1),
         label="meas norm", linestyle="--")
# plt.ylim(-10,40)
plt.legend()


# %%

# # acc.shape, len(acc_time)
# meas_time = np.array(meas_time)

# interpolated_meas_time = np.interp(acc_time, meas_time, meas_time)
# interpolated_meas_time.shape, len(acc_time)

# # interpolated_meas = np.interp(acc_time, times, new_meas[:,0])
# # plt.plot(interpolated_meas)

# interpolated_gt = np.interp(acc_time, gt_time, gt_pos[:, 0])
# plt.plot(interpolated_gt)

# i need measurements of P and measurements of acc with same timestamps
# meas_pos.shape, acc.shape

# interpolate meas time to match acc time
# interpolated_meas_time = np.interp(acc_time, meas_time, meas_time)
interpolated_meas_x = np.interp(acc_time, meas_time, meas_pos[:, 0])
interpolated_meas_y = np.interp(acc_time, meas_time, meas_pos[:, 1])
interpolated_meas_z = np.interp(acc_time, meas_time, meas_pos[:, 2])
interpolated_meas = np.stack(
    [interpolated_meas_x, interpolated_meas_y, interpolated_meas_z], axis=1)
# sensor data ready

# # moving on to ground truth
# # interpolate also with the copy
# # interpolated_gt_time = np.interp(acc_time, gt_time, gt_time)
# interpolated_gt_x = np.interp(acc_time, gt_time, gt_pos[:, 0])
# interpolated_gt_y = np.interp(acc_time, gt_time, gt_pos[:, 1])
# interpolated_gt_z = np.interp(acc_time, gt_time, gt_pos[:, 2])
# interpolated_gt = np.stack(
#     [interpolated_gt_x, interpolated_gt_y, interpolated_gt_z], axis=1)

# %%


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
    vel = savgol_filter(vel, 200, 3)
    delta2_pose = np.diff(vel)
    acc = delta2_pose / delta_time[1:]
    return vel, acc


# interpolate gt position
interpolated_gt_x = interpolate_gt_pos(gt_time, -gt_pos[:, 0])
interpolated_gt_y = interpolate_gt_pos(gt_time, -gt_pos[:, 1])
interpolated_gt_z = interpolate_gt_pos(gt_time, gt_pos[:, 2])

gt_vel_x, gt_acc_x = get_vel_and_acc(gt_time, -gt_pos[:, 0])
gt_vel_y, gt_acc_y = get_vel_and_acc(gt_time, -gt_pos[:, 1])
gt_vel_z, gt_acc_z = get_vel_and_acc(gt_time, gt_pos[:, 2])

gt_pos = np.stack([interpolated_gt_x, interpolated_gt_y,
                  interpolated_gt_z], axis=1)
gt_vel = np.stack([gt_vel_x, gt_vel_y, gt_vel_z], axis=1)
gt_acc = np.stack([gt_acc_x, gt_acc_y, gt_acc_z], axis=1)

plt.figure(dpi=300)
plt.plot(acc_time[:-1], gt_vel_y, label="gt vel x")
plt.plot(acc_time[:-2], gt_acc_y, label="gt acc x")

# %%

plt.plot(acc_time, gt_pos, label=["x", "y", "z"])
plt.plot(acc_time, interpolated_meas, linestyle="--", label=["x", "y", "z"])
plt.legend()

# %%


def get_F():
    return torch.tensor(Ad, dtype=torch.double)


def get_H():
    return torch.tensor(H, dtype=torch.double)


def initial_observation_to_state(z):
    # z = [p,a]
    x = torch.zeros(9, dtype=torch.double)
    x[:3] = torch.tensor(gt_pos[0, :3], dtype=torch.double)
    # x[:3] = z[:3]
    # x[6:] = z[3:]
    return x


def loss_fun():
    return lambda pred, x: ((pred[:3]-x[:3])**2).sum()


def model_args():
    return dict(
        dim_x=9,
        dim_z=6,
        init_z2x=initial_observation_to_state,
        F=get_F(),
        H=get_H(),
        loss_fun=loss_fun(),
    )


# %%

# X = [np.vstack((p,v,a)).astype(np.float64).T] * 75
# Z = [np.vstack((yp, ya)).astype(np.float64).T] * 75

X = [np.hstack((gt_pos[:-2, :], gt_vel[:-1, :], gt_acc)
               ).astype(np.float64)]
Z = [np.hstack((interpolated_meas[:-2], -acc[:-2])).astype(np.float64)]

# X = []
# Z = []
# # split timestamps into intervals of 1000 elements
# for i in range(0, len(acc_time), 1000):
#     if i+1000 > len(acc_time):
#         break
#     for i in range(7):
#         X.append(np.hstack((gt_pos[i:i+1000, :], gt_vel[i:i+1000, :], gt_acc[i:i+1000, :])).astype(np.float64))
#         Z.append(np.hstack((interpolated_meas[i:i+1000], acc[i:i+1000])).astype(np.float64))

# %%

# Define model
okf_model_args = model_args()
print('---------------\nModel arguments:\n', okf_model_args)
model = okf.OKF(**okf_model_args, optimize=True, model_name='OKF_REAL')

# %%


def print_for_copy(np_array):
    print("np.array([", end="")
    for row in np_array:
        print("[", end="")
        for el in row:
            print(f"{el:.7f},", end="")
        print("],")
    print("])")

# Run training


print_for_copy(model.get_Q())
print_for_copy(model.get_R())

res, _ = okf.train(model, Z, X, verbose=1, n_epochs=5,
                   batch_size=1, to_save=False)

print(res)

print_for_copy(model.get_Q())
print_for_copy(model.get_R())
# %%
