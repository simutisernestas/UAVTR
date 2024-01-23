# %%
from scipy.signal import savgol_filter
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import okf
import scipy
os.makedirs('models', exist_ok=True)

STATE_TIME_COLUMN = 0
SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
BAGS_LIST = [
    '18_0',
    'latest_flight_mode0',
    'latest_flight_mode1',
    'latest_flight_mode2',
]
NTH_FROM_BACK = 1
LIVE = len(sys.argv) == 1 or not sys.argv[1].isdigit()
PLOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

if not LIVE:
    BAG_NAME = BAGS_LIST[int(sys.argv[1])]
else:
    BAG_NAME = BAGS_LIST[0]


def load_latest(name, shape):
    latest_file = sorted([f for f in os.listdir(
        SAVE_DIR) if f'{BAG_NAME}_{name}' in f])[-NTH_FROM_BACK]
    print(f'latest {name} file: ', latest_file)
    return np.load(f'{SAVE_DIR}/{latest_file}').reshape(-1, shape)


state_data = load_latest("state_data", 17+3+3)
attitude_state_data = load_latest("attitude_state", 4)
attitude_state_time = attitude_state_data[:, 0]
attitude_px4_data = load_latest("attitude_px4", 4)
attitude_px4_time = attitude_px4_data[:, 0]
vel_measurements_data = load_latest("vel_measurements", 4)
vel_measurements_time = vel_measurements_data[:, 0]
pos_measurements_data = load_latest("pos_measurements", 4)
pos_measurements_time = pos_measurements_data[:, 0]
imu_data = load_latest("imu_data", 4)
imu_time = imu_data[:, 0]

GT_NAME = BAG_NAME if "mode" not in BAG_NAME else "_".join(
    BAG_NAME.split('_')[:-1])
gt_data = np.load(f'{SAVE_DIR}/{GT_NAME}_gt.npz')

drone_time = gt_data['drone_time']
drone_pos = gt_data['drone_pos']
drone_vel = gt_data['drone_vel']
boat_time = gt_data['boat_time']
boat_pos = gt_data['boat_pos']

# cap boat data to match drone data
boat_data_start = np.argmin(np.abs(boat_time - drone_time[0]))
boat_data_end = np.argmin(np.abs(boat_time - drone_time[-1]))
boat_time = boat_time[boat_data_start:boat_data_end]
boat_pos = boat_pos[boat_data_start:boat_data_end, :]

state_time = state_data[:, STATE_TIME_COLUMN]
# remove everything before KF initialization
state_non_zero = np.abs(state_data[:, 1]) > 0
state_data = state_data[state_non_zero, :]
state_time = state_time[state_non_zero]
if BAG_NAME != '18_0':
    state_time -= 105

# Define a tolerance level
tolerance = 0.09
# Calculate the absolute difference between all pairs of timestamps
diffs = np.abs(drone_time[:, None] - boat_time)
# Find where the difference is less than the tolerance
matching_indices = np.where(diffs < tolerance)
# Get the matching timestamps from drone_time
matching_timestamps = drone_time[matching_indices[0]]
drone_time = drone_time[matching_indices[0]]
boat_time = boat_time[matching_indices[1]]
drone_pos = drone_pos[matching_indices[0], :]
boat_pos = boat_pos[matching_indices[1], :]
drone_vel = drone_vel[matching_indices[0], :]

print('drone_time.shape: ', drone_time.shape)
print('drone_pos.shape: ', drone_pos.shape)
print('drone_vel.shape: ', drone_vel.shape)
print('boat_time.shape: ', boat_time.shape)
print('boat_pos.shape: ', boat_pos.shape)
print('state_data.shape: ', state_data.shape)

if BAG_NAME == BAGS_LIST[0]:
    t0 = np.argmin(np.abs(500 - boat_time))
    boat_pos_var = np.var(boat_pos[t0:, :], axis=0)
    boat_pos_std = np.sqrt(boat_pos_var)
    groundtruth_3std = boat_pos_std*3*2
    static_boat_pos = np.mean(boat_pos[t0:, :], axis=0)
    print("3 std: ", groundtruth_3std)
    relative_pos_gt = static_boat_pos - drone_pos[:boat_pos.shape[0], :]
else:
    raise NotImplementedError("TODO:")

def take_diff(t, pos, interp_time=None):
    # smooth position
    if interp_time is None:
        interp_time = t
        interpolated_pos = pos
    else:
        interp_F = interpolate.interp1d(
            t, pos, kind='linear', fill_value="extrapolate")
        interpolated_pos = interp_F(interp_time)
    delta_time = np.diff(interp_time)
    delta_pose = np.diff(interpolated_pos)
    deriv = delta_pose / delta_time
    # deriv = savgol_filter(deriv, 10, 3)
    return deriv


# # acc will be in NED, but i need ENU
# gt_acc_x = take_diff(drone_time, drone_vel[:, 1], imu_time)
# gt_acc_y = take_diff(drone_time, drone_vel[:, 0], imu_time)
# gt_acc_z = take_diff(drone_time, drone_vel[:, 2], imu_time)
# gt_acc = np.vstack((gt_acc_x, gt_acc_y, gt_acc_z)).T
# gt_acc.shape
# # plt.plot(imu_time[:-1], gt_acc_x, label='gt acc x')
# # plt.plot(imu_time, imu_data[:, 1], label='gt vel x', alpha=0.1)
# # plt.legend()

# t0_idx = np.argmin(np.abs(drone_time - vel_measurements_time[0]))
# t1_idx = np.argmin(np.abs(drone_time - vel_measurements_time[-1]))

# vel_measurements_data = vel_measurements_data[
#     (vel_measurements_data[:, 1] != 0) &
#     (vel_measurements_data[:, 2] != 0) &
#     (vel_measurements_data[:, 3] != 0), :]
# vel_measurements_time = vel_measurements_data[:, 0]

# plt.figure(figsize=(10, 5))
# plt.plot(drone_time[t0_idx:t1_idx],
#          drone_vel[t0_idx:t1_idx, 0], label='drone n')
# plt.plot(vel_measurements_time, -vel_measurements_data[:, 2], label='vel e')
# plt.figure(figsize=(10, 5))
# plt.plot(drone_time[t0_idx:t1_idx],
#          drone_vel[t0_idx:t1_idx, 1], label='drone e')
# plt.plot(vel_measurements_time, -vel_measurements_data[:, 1], label='vel n')
# plt.figure(figsize=(10, 5))
# plt.plot(drone_time[t0_idx:t1_idx],
#          drone_vel[t0_idx:t1_idx, 2], label='drone d')
# plt.plot(vel_measurements_time, vel_measurements_data[:, 3], label='vel d')
# plt.legend()

# t0_idx = np.argmin(np.abs(drone_time - pos_measurements_time[0]))
# t1_idx = np.argmin(np.abs(drone_time - pos_measurements_time[-1]))

# plt.figure(figsize=(10, 5))
# plt.plot(drone_time[t0_idx:t1_idx],
#          relative_pos_gt[t0_idx:t1_idx, 0], label='drone n')
# plt.plot(pos_measurements_time, pos_measurements_data[:, 1], label='pos n')
# plt.figure(figsize=(10, 5))
# plt.plot(drone_time[t0_idx:t1_idx],
#          relative_pos_gt[t0_idx:t1_idx, 1], label='drone e')
# plt.plot(pos_measurements_time, pos_measurements_data[:, 2], label='pos e')
# plt.figure(figsize=(10, 5))
# plt.plot(drone_time[t0_idx:t1_idx],
#          relative_pos_gt[t0_idx:t1_idx, 2], label='drone d')
# plt.plot(pos_measurements_time, pos_measurements_data[:, 3], label='pos d')

# %%


def interp_3d(time, data, interp_time):
    interp_data = np.zeros((interp_time.shape[0], 3))
    for i in range(3):
        interp_F = interpolate.interp1d(
            time, data[:, i],
            kind='linear',
            fill_value="extrapolate")
        interp_data[:, i] = interp_F(interp_time)
    return interp_data


t0_idx = np.argmin(np.abs(imu_time - pos_measurements_time[0])) + 1
t1_idx = np.argmin(np.abs(imu_time - 500))
# t1_idx = np.argmin(np.abs(imu_time - 417.0))

pos_meas = interp_3d(pos_measurements_time,
                     pos_measurements_data[:, 1:],
                     imu_time[t0_idx:t1_idx])
pos_gt = interp_3d(drone_time,
                   relative_pos_gt,
                   imu_time[t0_idx:t1_idx])

plt.plot(imu_time[t0_idx:t1_idx], pos_meas[:, 0], label='n')
plt.plot(imu_time[t0_idx:t1_idx], pos_meas[:, 1], label='e')
plt.plot(imu_time[t0_idx:t1_idx], pos_meas[:, 2], label='d')
plt.plot(imu_time[t0_idx:t1_idx], pos_gt[:, 0], label='n gt')
plt.plot(imu_time[t0_idx:t1_idx], pos_gt[:, 1], label='e gt')
plt.plot(imu_time[t0_idx:t1_idx], pos_gt[:, 2], label='d gt')
plt.legend()

# %%

vel_meas = interp_3d(vel_measurements_time,
                     vel_measurements_data[:, 1:], imu_time[t0_idx:t1_idx])

drone_vel_ = drone_vel.copy()
drone_vel_[:, [0, 1]] = drone_vel_[:, [1, 0]]

vel_gt = interp_3d(drone_time,
                   drone_vel_,
                   imu_time[t0_idx:t1_idx])

plt.figure()
plt.plot(imu_time[t0_idx:t1_idx], vel_meas[:, 0], label='n')
plt.plot(imu_time[t0_idx:t1_idx], vel_meas[:, 1], label='e')
plt.plot(imu_time[t0_idx:t1_idx], vel_meas[:, 2], label='d')
plt.plot(imu_time[t0_idx:t1_idx], vel_gt[:, 0], label='n gt')
plt.plot(imu_time[t0_idx:t1_idx], vel_gt[:, 1], label='e gt')
plt.plot(imu_time[t0_idx:t1_idx], vel_gt[:, 2], label='d gt')
plt.ylim([-2, 2])
plt.legend()

# %%

gt_acc_x = take_diff(imu_time[t0_idx:t1_idx],
                     vel_gt[:, 0], imu_time[t0_idx:t1_idx+1])
gt_acc_y = take_diff(imu_time[t0_idx:t1_idx],
                     vel_gt[:, 1], imu_time[t0_idx:t1_idx+1])
gt_acc_z = take_diff(imu_time[t0_idx:t1_idx],
                     vel_gt[:, 2], imu_time[t0_idx:t1_idx+1])
gt_acc = np.vstack((gt_acc_x, gt_acc_y, gt_acc_z)).T

acc_gt = gt_acc[:]
acc_meas = imu_data[t0_idx:t1_idx, 1:]

# plot
plt.figure()
plt.plot(imu_time[t0_idx:t1_idx], acc_meas[:, 0], label='n')
plt.plot(imu_time[t0_idx:t1_idx], acc_meas[:, 1], label='e')
# plt.plot(imu_time[t0_idx:t1_idx], acc_meas[:, 2], label='d')
plt.plot(imu_time[t0_idx:t1_idx], acc_gt[:, 0], label='n gt')
plt.plot(imu_time[t0_idx:t1_idx], acc_gt[:, 1], label='e gt')
# plt.plot(imu_time[t0_idx:t1_idx], acc_gt[:, 2], label='d gt')
plt.legend()
# plt.ylim([-4, 4])

# %%

vel_boat_gt = np.zeros((vel_gt.shape[0], 2))
acc_bias_gt = np.zeros((acc_gt.shape[0], 3))
all_state = np.hstack((pos_gt, vel_gt, vel_boat_gt,
                      acc_gt, acc_bias_gt), dtype=np.float64)
all_meas = np.hstack((pos_meas, vel_meas, acc_meas), dtype=np.float64)
X = [all_state]
Z = [all_meas]
# X = []
# Z = []
# # divide into 200 samples
# for i in range(0, all_state.shape[0], 200):
#     X.append(all_state[i:i+200, :])
#     Z.append(all_meas[i:i+200, :])

# %%

N_STATES = 14


def get_F():
    A = np.zeros((N_STATES, N_STATES))
    A[0:3, 3:6] = -np.eye(3)
    A[0:2, 6:8] = np.eye(2)
    A[3:6, 8:11] = np.eye(3)
    dt = 1.0/128.0
    F = scipy.linalg.expm(A*dt)
    return torch.tensor(F, dtype=torch.double)


def get_H():
    C = np.zeros((9, N_STATES))
    C[:3, :3] = np.eye(3)
    C[3:6, 3:6] = np.eye(3)
    C[6:9, 8:11] = np.eye(3)
    C[6:9, 11:14] = np.eye(3)
    return torch.tensor(C, dtype=torch.double)


def initial_observation_to_state(z):
    x = torch.zeros((N_STATES,), dtype=torch.double)
    x[:3] = z[:3]
    x[3:6] = z[3:6]
    x[8:11] = z[6:9]
    return x


def loss_fun():
    return lambda pred, x: ((pred-x)**2).sum()


def model_args():
    return dict(
        dim_x=N_STATES,
        dim_z=9,
        init_z2x=initial_observation_to_state,
        F=get_F(),
        H=get_H(),
        loss_fun=loss_fun(),
    )

# %%


# Define model
okf_model_args = model_args()
print('---------------\nModel arguments:\n', okf_model_args)
model = okf.OKF(**okf_model_args, optimize=True, model_name='OKF_REAL')
RESET = False
if not RESET:
    model.load_model(fname='OKF_REAL.m', base_path='models')

# %%

res, _ = okf.train(model, Z, X, verbose=1, n_epochs=500, lr=1e-3,
                   batch_size=1, to_save=True, lr_decay_freq=200,
                   noise_estimation_initialization=RESET,
                   reset_model=RESET)

# %%

np.set_printoptions(precision=8, suppress=True)
print(f"Q = {list(model.get_Q().reshape(-1))}")

R = model.get_R()
pos_R = R[:2, :2]
vel_R = R[3:6, 3:6]
acc_R = R[6:, 6:]

print(f"pos_R = {list(pos_R.reshape(-1))}")
print(f"vel_R = {list(vel_R.reshape(-1))}")
print(f"acc_R = {list(acc_R.reshape(-1))}")
print(f"Heigh variance: {R[2,2]}")

np.diag(model.get_Q())
