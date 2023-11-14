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
np.set_printoptions(precision=6, suppress=True)
from scipy.signal import butter, lfilter


# %%

# read cvs file record.csv
df = pd.read_csv('real.csv')
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
est_pos = []
est_time = []
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
    if not np.isnan(row["/cam_target_pos/header/stamp"]):
        pos = [row["/cam_target_pos/point/x"],
               row["/cam_target_pos/point/y"],
               row["/cam_target_pos/point/z"]]
        est_pos.append(pos)
        est_time.append(row["/cam_target_pos/header/stamp"])

# interpolate gt_pos to match meas_pos
gt_pos = np.array(gt_pos)
meas_pos = np.array(meas_pos)
acc = np.array(acc)
acc_time = np.array(acc_time)
gt_time = np.array(gt_time)
meas_time = np.array(meas_time)
est_pos = np.array(est_pos)
est_time = np.array(est_time)
print(len(gt_pos), len(meas_pos), len(acc))


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#%%
est_pos

# %%


plt.scatter(gt_time, gt_pos[:, 0], label="X gt")
plt.scatter(est_time, est_pos[:, 0], label="X meas")
plt.legend()
plt.figure()
plt.scatter(gt_time, gt_pos[:, 1], label="Y gt")
plt.scatter(est_time, est_pos[:, 1], label="Y meas")
plt.legend()
plt.figure()
plt.scatter(gt_time, gt_pos[:, 2], label="Z gt")
plt.scatter(est_time, est_pos[:, 2], label="Z meas")
plt.legend()
# plt.show()
# plot norms
plt.scatter(gt_time, np.linalg.norm(gt_pos, axis=1), label="gt")
plt.scatter(est_time, np.linalg.norm(est_pos, axis=1), label="meas")
plt.legend()
plt.figure()

#%%

filter_order = 2
cutoff_frequency = 1
b, a = butter(filter_order, cutoff_frequency,
                fs=200, btype="lowpass", analog=False)

filtered_acc_x = lfilter(b, a, acc[:, 0])
filtered_acc_y = lfilter(b, a, acc[:, 1])
filtered_acc_z = lfilter(b, a, acc[:, 2])
acc = np.stack([filtered_acc_x, filtered_acc_y, filtered_acc_z], axis=1)

plt.plot(acc)

# N = 2
# AXIS = 2
# plt.scatter(acc_time[:-(N-1)],
#             moving_average(acc[:, AXIS], n=N), label="acc", s=.1)
# plt.scatter(gt_time, gt_pos[:, AXIS] / 20, label="X pos gt", s=.1)
# plt.legend()


# %%

np.unique(np.diff(acc_time), return_counts=True)

# %%

# x = [p3,v3,a3] 9 states in total
# u = []
# y = [p3,a3] 6 measurements in total

# A = [[0, 1, 0],
#      [0, 0, 1],
#      [0, 0, 0]]
N_STATES = 9
A = np.block([[np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
              [np.zeros((3, 3)), np.zeros((3, 3)), -np.eye(3)],
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
print(dt**2/2)


def get_F(dt):
    return np.block([[np.eye(3), dt*np.eye(3), -dt**2/2*np.eye(3)],
                     [np.zeros((3, 3)), np.eye(3), -dt*np.eye(3)],
                     [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)]])


assert (Ad == get_F(dt)).all()

# %%

# Define the system matrices
F = Ad

H = np.array(C)

# Q = np.eye(9)
# Q[:3, :3] *= 5
# Q[3:6, 3:6] *= 25
# Q[6:, 6:] *= 2
# R = np.eye(6)
# R[:3, :3] *= 1
# R[3:, 3:] *= 1
Q = np.array([[0.0000143,0.0000074,0.0000016,0.0000001,-0.0000001,-0.0000001,-0.0000028,0.0000220,0.0000218,],
[0.0000074,0.0000102,0.0000011,0.0000001,0.0000000,-0.0000000,-0.0000174,0.0000199,0.0000023,],
[0.0000016,0.0000011,0.0000030,0.0000001,-0.0000000,-0.0000000,-0.0000085,0.0000051,0.0000014,],
[0.0000001,0.0000001,0.0000001,0.0000052,0.0000020,0.0000003,-0.0006218,-0.0002983,-0.0000428,],
[-0.0000001,0.0000000,-0.0000000,0.0000020,0.0000044,0.0000006,-0.0002984,-0.0004917,-0.0000827,],
[-0.0000001,-0.0000000,-0.0000000,0.0000003,0.0000006,0.0000029,-0.0000425,-0.0000822,-0.0002806,],
[-0.0000028,-0.0000174,-0.0000085,-0.0006218,-0.0002984,-0.0000425,0.1594761,0.0763861,0.0109084,],
[0.0000220,0.0000199,0.0000051,-0.0002983,-0.0004917,-0.0000822,0.0763861,0.1260474,0.0210887,],
[0.0000218,0.0000023,0.0000014,-0.0000428,-0.0000827,-0.0002806,0.0109084,0.0210887,0.0720185,],
])
R = np.array([[17.3425122,11.5436677,2.1551933,0.0472867,0.0922534,-0.0238766,],
[11.5436677,16.8411307,2.6780333,0.2360581,0.0377527,-0.0631562,],
[2.1551933,2.6780333,2.1660042,0.2607522,0.0661012,-0.0224623,],
[0.0472867,0.2360581,0.2607522,0.2605860,0.0667360,0.0160195,],
[0.0922534,0.0377527,0.0661012,0.0667360,0.2314712,0.0447116,],
[-0.0238766,-0.0631562,-0.0224623,0.0160195,0.0447116,0.1799458,],
])

# Initial state estimate
x_hat = np.zeros((9, 1))
x_hat[:3] = meas_pos[0].reshape(3, 1)

# Initial error covariance
P = np.eye(9) * 10000

# data prep
# print(len(meas_pos), len(acc))
first_acc_index = np.searchsorted(acc_time, meas_time[0])
print(first_acc_index)
last_pos_meas_index = 0
timestamp = acc_time[first_acc_index-1]

record = []
# Kalman Filter loop
for i in range(acc[first_acc_index:].shape[0]):
    dt = acc_time[first_acc_index+i] - timestamp
    timestamp = acc_time[first_acc_index+i]
    F = get_F(dt)

    # Predict step
    x_hat = F @ x_hat
    P = F @ P @ F.T + Q

    p_meas = None
    acc_meas_time = acc_time[first_acc_index+i]
    if last_pos_meas_index == len(meas_time):
        print(f"Break on: {i}")
        break
    if acc_meas_time > meas_time[last_pos_meas_index]:
        p_meas = meas_pos[last_pos_meas_index].reshape(3, 1)
        last_pos_meas_index += 1

    # Position update
    if p_meas is not None:
        # p_meas = meas_pos[last_pos_meas_index].reshape(3, 1)
        Hp = H[:3, :]
        K = P @ Hp.T @ np.linalg.inv(Hp @ P @ Hp.T + R[:3, :3])
        x_hat = x_hat + K @ (p_meas - Hp @ x_hat)
        P = (np.eye(N_STATES) - K @ Hp) @ P

    # Acceleration update
    Ha = H[3:, :]
    K = P @ Ha.T @ np.linalg.inv(Ha @ P @ Ha.T + R[3:, 3:])
    x_hat = x_hat + K @ (acc[i].reshape(3, 1) - Ha @ x_hat)
    P = (np.eye(N_STATES) - K @ Ha) @ P

    record.append(x_hat.T)

print(P)

record = np.array(record).reshape(-1, 9)
plt.figure(dpi=200)  # norm
plt.plot(acc_time[first_acc_index:first_acc_index+len(record)], np.linalg.norm(
    record[:, :3], axis=1), label="KF norm")
plt.plot(gt_time, np.linalg.norm(gt_pos, axis=1), label="gt norm")
plt.plot(meas_time, np.linalg.norm(meas_pos, axis=1),
         label="meas norm", linestyle="--")
# plt.ylim(-10,40)
plt.legend()
plt.show()

# plot axes of estimation
plt.figure(dpi=200)
plt.plot(acc_time[first_acc_index:first_acc_index+len(record)],
         record[:, 0], label="KF X")
plt.plot(acc_time[first_acc_index:first_acc_index+len(record)],
            record[:, 1], label="KF Y")
plt.plot(acc_time[first_acc_index:first_acc_index+len(record)],
            record[:, 2], label="KF Z")
plt.legend()
plt.show()


# %%

interpolated_meas_x = np.interp(acc_time, meas_time, meas_pos[:, 0])
interpolated_meas_y = np.interp(acc_time, meas_time, meas_pos[:, 1])
interpolated_meas_z = np.interp(acc_time, meas_time, meas_pos[:, 2])
interpolated_meas = np.stack(
    [interpolated_meas_x, interpolated_meas_y, interpolated_meas_z], axis=1)


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
interpolated_gt_x = interpolate_gt_pos(gt_time, gt_pos[:, 0])
interpolated_gt_y = interpolate_gt_pos(gt_time, gt_pos[:, 1])
interpolated_gt_z = interpolate_gt_pos(gt_time, gt_pos[:, 2])

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
    return torch.tensor(H, dtype=torch.double)


def initial_observation_to_state(z):
    # z = [p,a]
    x = torch.zeros(9, dtype=torch.double)
    x[:3] = z[:3]
    x[6:] = z[3:]
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

X = [np.hstack((interpolated_gt_pos[:-2, :],
               gt_vel[:-1, :], -gt_acc)).astype(np.float64)]
Z = [np.hstack((interpolated_meas[:-2], acc[:-2])).astype(np.float64)]

# %%

# Define model
okf_model_args = model_args()
print('---------------\nModel arguments:\n', okf_model_args)
model = okf.OKF(**okf_model_args, optimize=True, model_name='OKF_REAL')

#%%

def print_for_copy(np_array, matrix):
    print(f"{matrix} = np.array([", end="")
    for row in np_array:
        print("[", end="")
        for el in row:
            print(f"{el:.7e},", end="")
        print("],")
    print("])")

model.estimate_noise(X,Z)
print_for_copy(model.get_Q(), "Q")
print_for_copy(model.get_R(), "R")

# %%

# Run training

print_for_copy(model.get_Q(), 'Q')
print_for_copy(model.get_R(), 'R')

# model.estimate_noise(X,Z)

res, _ = okf.train(model, Z, X, verbose=1, n_epochs=0,
                   batch_size=1, to_save=False)

# print(res)

print_for_copy(model.get_Q(), 'Q')
print_for_copy(model.get_R(), 'R')
# %%
