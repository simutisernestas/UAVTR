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
    if not np.isnan(row["/gt_target_pos/point/x"]):
        # dlat = row["/gps_postproc/latitude"]
        # dlng = row["/gps_postproc/longitude"]
        # dalt = row["/gps_postproc/altitude"]
        # time = row["/gps_postproc/header/stamp"]
        # pos = pm.geodetic2enu(lat, lng, alt, dlat, dlng, dalt)
        gt_pos.append([
            row["/gt_target_pos/point/x"],
            row["/gt_target_pos/point/y"],
            row["/gt_target_pos/point/z"]
        ])
        time = row["/gt_target_pos/header/stamp"]
        gt_time.append(time)
    if not np.isnan(row["/vec_target/points.1/x"]):
        pos = (row["/vec_target/points.1/x"],
               row["/vec_target/points.1/y"],
               row["/vec_target/points.1/z"])
        meas_pos.append(pos)
        meas_time.append(
            row["/vec_target/header/stamp"])
    if not np.isnan(row["/imu/data_world/header/stamp"]):
        acc.append([
            row["/imu/data_world/vector/x"],
            row["/imu/data_world/vector/y"],
            row["/imu/data_world/vector/z"]
        ])
        acc_time.append(
            row["/imu/data_world/header/stamp"])

# interpolate gt_pos to match meas_pos
gt_pos = np.array(gt_pos)
meas_pos = np.array(meas_pos)
acc = np.array(acc)
print(len(gt_pos), len(meas_pos), len(acc))


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

# filter_order = 2
# cutoff_frequency = 1
# b, a = butter(filter_order, cutoff_frequency,
#               fs=200, btype="lowpass", analog=False)

# filtered_acc_x = lfilter(b, a, acc[:, 0])
# filtered_acc_y = lfilter(b, a, acc[:, 1])
# filtered_acc_z = lfilter(b, a, acc[:, 2])
# acc = np.stack([filtered_acc_x, filtered_acc_y, filtered_acc_z], axis=1)
# plt.plot(acc)

# %%

N = 2
AXIS = 0
plt.scatter(acc_time[:-(N-1)],
            moving_average(acc[:, AXIS], n=N), label="acc", s=.1)
# plt.scatter(gt_time, gt_pos[:, AXIS] / 20, label="X pos gt", s=.1)
plt.legend()


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

Q = np.eye(9)
Q[:3, :3] *= 5
Q[3:6, 3:6] *= 25
Q[6:, 6:] *= 2
R = np.eye(6)
R[:3, :3] *= 1
R[3:, 3:] *= 1
Q = np.array([[0.0002730, -0.0730942, 0.0015254, 0.0010283, 0.0105967, -0.0003864, 0.0005567, -0.0009134, 0.0004319,],
              [-0.0730942, 19.5677683, -0.4085797, -0.2752198, -2.8365703,
                  0.1026148, -0.1485698, 0.2443645, -0.1165228,],
              [0.0015254, -0.4085797, 0.1235749, -0.0283929, -0.0719825,
                  0.4465410, -0.2375252, 0.0858468, 0.4884473,],
              [0.0010283, -0.2752198, -0.0283929, 0.3074789, 0.0113581,
               0.5610426, -0.0368077, 0.0247507, 0.1010197,],
              [0.0105967, -2.8365703, -0.0719825, 0.0113581, 0.6296948, -
               0.6627772, 0.5223519, -0.6293901, -0.6781707,],
              [-0.0003864, 0.1026148, 0.4465410, 0.5610426, -0.6627772,
               4.7292942, -1.5438142, 1.0392190, 2.0104655,],
              [0.0005567, -0.1485698, -0.2375252, -0.0368077, 0.5223519, -
               1.5438142, 1.4514090, -2.3024521, -1.2260295,],
              [-0.0009134, 0.2443645, 0.0858468, 0.0247507, -0.6293901,
               1.0392190, -2.3024521, 5.2198419, 0.1918878,],
              [0.0004319, -0.1165228, 0.4884473, 0.1010197, -0.6781707,
               2.0104655, -1.2260295, 0.1918878, 5.2376935,],
              ])
R = np.array([[71.1221847, -10.6691912, -7.8696962, -48.8067143, -23.8458910, -16.3731660,],
              [-10.6691912, 4.8313547, -0.0089825,
                  5.7083177, 4.9832018, 3.4255008,],
              [-7.8696962, -0.0089825, 1.4453015,
                  5.8237033, 1.9522835, 1.9565010,],
              [-48.8067143, 5.7083177, 5.8237033,
                  34.5160381, 15.9204568, 10.1253587,],
              [-23.8458910, 4.9832018, 1.9522835,
                  15.9204568, 9.4214830, 5.3278395,],
              [-16.3731660, 3.4255008, 1.9565010,
                  10.1253587, 5.3278395, 5.9373489,],
              ])
# Initial state estimate
x_hat = np.zeros((9, 1))
x_hat[:3] = gt_pos[0].reshape(3, 1)

# Initial error covariance
P = np.eye(9) * 1000

# data prep
# print(len(meas_pos), len(acc))
first_acc_index = np.searchsorted(acc_time, meas_time[0])
print(first_acc_index)
last_pos_meas_index = 0
timestamp = acc_time[first_acc_index-1]
counter = 0
record = []
# Kalman Filter loop
for i in range(acc[first_acc_index:].shape[0]):
    dt = acc_time[first_acc_index+i] - timestamp
    timestamp = acc_time[first_acc_index+i]
    if dt > 0.0:
        F = get_F(dt)
    else:
        continue

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

    counter += 1
    # Position update
    if p_meas is not None:  # and counter % 2 != 0:
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
# plt.show()

# plot axes of estimation
plt.figure(dpi=200)
plt.plot(acc_time[first_acc_index:first_acc_index+len(record)],
         record[:, 0], label="KF X")
plt.plot(acc_time[first_acc_index:first_acc_index+len(record)],
         record[:, 1], label="KF Y")
plt.plot(acc_time[first_acc_index:first_acc_index+len(record)],
         record[:, 2], label="KF Z")
plt.legend()
# plt.show()


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

# plt.figure(dpi=300)
# plt.scatter(acc_time, acc[:, 0], label="real", s=.1)
# plt.scatter(acc_time[:-2], gt_acc_x, label="GT", s=.1)
# plt.legend()
# plt.show()

plt.plot(acc_time, interpolated_gt_pos, label="gt")
plt.plot(acc_time, interpolated_meas, linestyle="--", label="meas")
# plt.show()

# %%

H.shape


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
    return lambda pred, x: ((pred-x)**2).sum()


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

# split into itervals of 1000
X = [
    np.array_split(x, x.shape[0]//200)
    for x in X
]
Z = [
    np.array_split(z, z.shape[0]//200)
    for z in Z
]
X = X[0]
Z = Z[0]
len(X), len(Z)

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
            print(f"{el:.7f},", end="")
        print("],")
    print("])")


# Run training
print_for_copy(model.get_Q(), 'Q')
print_for_copy(model.get_R(), 'R')

# model.estimate_noise(X,Z)

# for i in range(30):
res, _ = okf.train(model, Z, X, verbose=1, n_epochs=10,
                   batch_size=1, to_save=False, p_valid=0.0, noise_estimation_initialization=True, lr=0.1)
print(res)

# %%
print_for_copy(model.get_Q(), 'Q')
print_for_copy(model.get_R(), 'R')
# %%
