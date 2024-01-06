# %%
from scipy.signal import savgol_filter
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import okf

STATE_TIME_COLUMN = 0
STATE_TARGET_IN_SIGHT_COLUMN = 13
STATE_COV_X_COLUMN = 14
STATE_COV_Y_COLUMN = 15
STATE_COV_Z_COLUMN = 16
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


state_data = load_latest("state_data", 17)
attitude_state_data = load_latest("attitude_state", 4)
attitude_state_time = attitude_state_data[:, 0]
attitude_px4_data = load_latest("attitude_px4", 4)
attitude_px4_time = attitude_px4_data[:, 0]
vel_measurements_data = load_latest("vel_measurements", 4)
vel_measurements_time = vel_measurements_data[:, 0]
pos_measurements_data = load_latest("pos_measurements", 4)
pos_measurements_time = pos_measurements_data[:, 0]

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

# binary signal indicating whether the target is in sight or not
target_in_sight = state_data[:, STATE_TARGET_IN_SIGHT_COLUMN]
binary_sight = target_in_sight > 0

# %%

t0_idx = np.argmin(np.abs(drone_time - vel_measurements_time[0]))
t1_idx = np.argmin(np.abs(drone_time - vel_measurements_time[-1]))

vel_measurements_data = vel_measurements_data[
    (vel_measurements_data[:, 1] != 0) &
    (vel_measurements_data[:, 2] != 0) &
    (vel_measurements_data[:, 3] != 0), :]
vel_measurements_time = vel_measurements_data[:, 0]

plt.figure(figsize=(10, 5))
plt.plot(drone_time[t0_idx:t1_idx],
         drone_vel[t0_idx:t1_idx, 0], label='drone n')
plt.plot(vel_measurements_time, -vel_measurements_data[:, 2], label='vel e')
plt.figure(figsize=(10, 5))
plt.plot(drone_time[t0_idx:t1_idx],
         drone_vel[t0_idx:t1_idx, 1], label='drone e')
plt.plot(vel_measurements_time, -vel_measurements_data[:, 1], label='vel n')
plt.figure(figsize=(10, 5))
plt.plot(drone_time[t0_idx:t1_idx],
         drone_vel[t0_idx:t1_idx, 2], label='drone d')
plt.plot(vel_measurements_time, vel_measurements_data[:, 3], label='vel d')
plt.legend()

# %%

t0_idx = np.argmin(np.abs(drone_time - pos_measurements_time[0]))
t1_idx = np.argmin(np.abs(drone_time - pos_measurements_time[-1]))

plt.figure(figsize=(10, 5))
plt.plot(drone_time[t0_idx:t1_idx],
         relative_pos_gt[t0_idx:t1_idx, 0], label='drone n')
plt.plot(pos_measurements_time, pos_measurements_data[:, 1], label='vel n')
plt.figure(figsize=(10, 5))
plt.plot(drone_time[t0_idx:t1_idx],
         relative_pos_gt[t0_idx:t1_idx, 1], label='drone e')
plt.plot(pos_measurements_time, pos_measurements_data[:, 2], label='vel e')
plt.figure(figsize=(10, 5))
plt.plot(drone_time[t0_idx:t1_idx],
         relative_pos_gt[t0_idx:t1_idx, 2], label='drone d')
plt.plot(pos_measurements_time, pos_measurements_data[:, 3], label='vel d')
plt.legend()


# %%

def interp_3d(time, data, interp_time):
    interp_data = np.zeros((interp_time.shape[0], 3))
    for i in range(3):
        interp_F = interpolate.interp1d(
            time, data[:, i],
            kind='cubic',
            fill_value="extrapolate")
        interp_data[:, i] = interp_F(interp_time)
    return interp_data


t0_idx = np.argmin(np.abs(drone_time - pos_measurements_time[0])) + 1
t1_idx = np.argmin(np.abs(drone_time - pos_measurements_time[-1]))
t1_idx = np.argmin(np.abs(drone_time - 417.0))

pos_meas = interp_3d(pos_measurements_time,
                     pos_measurements_data[:, 1:], drone_time[t0_idx:t1_idx])

plt.plot(drone_time[t0_idx:t1_idx], pos_meas[:, 0], label='n')
plt.plot(drone_time[t0_idx:t1_idx], pos_meas[:, 1], label='e')
plt.plot(drone_time[t0_idx:t1_idx], pos_meas[:, 2], label='d')
plt.plot(drone_time[t0_idx:t1_idx],
         relative_pos_gt[t0_idx:t1_idx, 0], label='n gt')
plt.plot(drone_time[t0_idx:t1_idx],
         relative_pos_gt[t0_idx:t1_idx, 1], label='e gt')
plt.plot(drone_time[t0_idx:t1_idx],
         relative_pos_gt[t0_idx:t1_idx, 2], label='d gt')
plt.legend()

# t0_idx = np.argmin(np.abs(drone_time - vel_measurements_time[0]))
# t1_idx = np.argmin(np.abs(drone_time - vel_measurements_time[-1]))

vel_meas = interp_3d(vel_measurements_time,
                     vel_measurements_data[:, 1:], drone_time[t0_idx:t1_idx])

drone_vel_ = drone_vel[t0_idx:t1_idx].copy()
drone_vel_[:, [0, 1]] = -drone_vel_[:, [1, 0]]

plt.figure()
plt.plot(drone_time[t0_idx:t1_idx], vel_meas[:, 0], label='n')
plt.plot(drone_time[t0_idx:t1_idx], vel_meas[:, 1], label='e')
plt.plot(drone_time[t0_idx:t1_idx], vel_meas[:, 2], label='d')
plt.plot(drone_time[t0_idx:t1_idx],
         drone_vel_[:, 0], label='n gt')
plt.plot(drone_time[t0_idx:t1_idx],
         drone_vel_[:, 1], label='e gt')
plt.plot(drone_time[t0_idx:t1_idx],
         drone_vel_[:, 2], label='d gt')
plt.ylim([-3, 3])
plt.legend()


# %%

# basically i need X and Z meaning ground truth and measurements stacked
# X gonna be relative_pos_gt and drone vel
# Z gonna be pos_measurements_data and vel_measurements_data


X = [np.hstack((relative_pos_gt[t0_idx:t1_idx],
               drone_vel_), dtype=np.float64)]
Z = [np.hstack((pos_meas, vel_meas), dtype=np.float64)]


def get_F():
    dt = 1.0 / 20.0
    Ad = np.eye(6)
    Ad[0, 3] = dt
    Ad[1, 4] = dt
    Ad[2, 5] = dt
    return torch.tensor(Ad, dtype=torch.double)


def get_H():
    C = np.eye(6)
    return torch.tensor(C, dtype=torch.double)


def initial_observation_to_state(z):
    return get_H() @ z


def loss_fun():
    return lambda pred, x: ((pred-x)**2).sum()


def model_args():
    return dict(
        dim_x=6,
        dim_z=6,
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


# %%

res, _ = okf.train(model, Z, X, verbose=1, n_epochs=100,
                   batch_size=1, to_save=False)

# TODO: i think i have to include the acceleration! for process noise matrix to be correct

# %%

model.get_Q(), model.get_R()
# print_for_copy(model.get_Q(), "Q")
# print_for_copy(model.get_R(), "R")


# %%
# def take_diff(t, pos, interp_time=None):
#     # smooth position
#     if interp_time is None:
#         interp_time = t
#         interpolated_pos = pos
#     else:
#         interp_F = interpolate.interp1d(
#             t, pos, kind='cubic', fill_value="extrapolate")
#         interpolated_pos = interp_F(interp_time)
#     delta_time = np.diff(interp_time)
#     delta_pose = np.diff(interpolated_pos)
#     vel = delta_pose / delta_time
#     vel = savgol_filter(vel, 10, 3)
#     return vel
# gt_acc_x = take_diff(drone_time, drone_vel[:, 0], drone_time)
# plt.plot(drone_time[:-1], gt_acc_x, label='gt acc x')
# plt.figure()
# plt.plot(drone_time, drone_vel[:, 0], label='gt vel x')
# plt.legend()
