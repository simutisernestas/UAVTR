# %%
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os

STATE_TIME_COLUMN = 0
STATE_TARGET_IN_SIGHT_COLUMN = -1
SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
BAG_NAME = '18_0'

latest_state_file = sorted([f for f in os.listdir(
    SAVE_DIR) if f'{BAG_NAME}_state_data' in f])[-1]
print('latest state file: ', latest_state_file)
# NpzFile 'gt.npz' with keys: drone_time, boat_time, drone_pos, boat_pos
gt_data = np.load(f'{SAVE_DIR}/{BAG_NAME}_gt.npz')
# load state estimation data from state_data.npy
state_data = np.load(f'{SAVE_DIR}/{latest_state_file}').reshape(-1, 14)

# load attitude estimation data from timstamp_bag_attitude_state.npy
latest_attitude_state_file = sorted([f for f in os.listdir(
    SAVE_DIR) if f'{BAG_NAME}_attitude_state' in f])[-1]
attitude_state_data = np.load(
    f'{SAVE_DIR}/{latest_attitude_state_file}').reshape(-1, 4)
print('attitude_state_data.shape: ', attitude_state_data.shape)
attitude_state_time = attitude_state_data[:, 0]

# load attitude px4 data from timstamp_bag_attitude_px4.npy
latest_attitude_px4_file = sorted([f for f in os.listdir(
    SAVE_DIR) if f'{BAG_NAME}_attitude_px4' in f])[-1]
attitude_px4_data = np.load(
    f'{SAVE_DIR}/{latest_attitude_px4_file}').reshape(-1, 4)
print('attitude_px4_data.shape: ', attitude_px4_data.shape)
attitude_px4_time = attitude_px4_data[:, 0]

drone_time = gt_data['drone_time']
boat_time = gt_data['boat_time']
drone_pos = gt_data['drone_pos']
boat_pos = gt_data['boat_pos']

# cap boat data to match drone data
boat_data_start = np.argmin(np.abs(boat_time - drone_time[0]))
boat_data_end = np.argmin(np.abs(boat_time - drone_time[-1]))
boat_time = boat_time[boat_data_start:boat_data_end]
boat_pos = boat_pos[boat_data_start:boat_data_end, :]

state_time = state_data[:, STATE_TIME_COLUMN]
target_in_sight = state_data[:, STATE_TARGET_IN_SIGHT_COLUMN]

# cap drone data to match the estimation
data_start = np.argmin(np.abs(drone_time - state_time[0]))
data_end = np.argmin(np.abs(drone_time - state_time[-1]))
drone_time = drone_time[data_start:data_end]
drone_pos = drone_pos[data_start:data_end, :]

# cap boat data to match the estimation
data_start = np.argmin(np.abs(boat_time - state_time[0]))
data_end = np.argmin(np.abs(boat_time - state_time[-1]))
boat_time = boat_time[data_start:data_end]
boat_pos = boat_pos[data_start:data_end, :]

# some residuals are still here
drone_pos = drone_pos[:boat_pos.shape[0], :]
drone_time = drone_time[:boat_pos.shape[0]]

print('drone_time.shape: ', drone_time.shape)
print('boat_time.shape: ', boat_time.shape)
print('drone_pos.shape: ', drone_pos.shape)
print('boat_pos.shape: ', boat_pos.shape)
print('state_data.shape: ', state_data.shape)

# relative grouthtruth position
relative_pos_gt = boat_pos - drone_pos[:boat_pos.shape[0], :]
# binary signal indicating whether the target is in sight or not
binary_sight = np.where(target_in_sight > 0)

# %%


def plot_data(t0_data, t1_data, state_data, state_index, pos_data, pos_index, est_label, gt_label, axis_label, binary_sight=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 5), dpi=200)
    for i, (state_idx, pos_idx, est_lbl, gt_lbl, axis_lbl) in enumerate(zip(state_index, pos_index, est_label, gt_label, axis_label)):
        axs[i].scatter(t0_data, state_data[:, state_idx],
                       label=est_lbl, s=1, marker='*')
        axs[i].scatter(t1_data, pos_data[:, pos_idx], label=gt_lbl, s=1)
        if binary_sight is not None:
            axs[i].scatter(t0_data[binary_sight], np.ones_like(t0_data)[
                           binary_sight], label='Target in FOV', s=1, color='green', marker='x')
        axs[i].legend(markerscale=5, loc='lower right')
        axs[i].grid(True, linestyle='-', linewidth=0.5)
        if i == 1:
            axs[i].set_ylabel(axis_lbl)
        elif i == 2:
            axs[i].set_xlabel(axis_lbl)
    fig.align_xlabels()
    fig.align_ylabels()
    fig.tight_layout()

# %%


plot_data(state_time, drone_time,
          state_data, [1, 2, 3],
          relative_pos_gt, [0, 1, 2],
          ['Estimation X', 'Estimation Y', 'Estimation Z'],
          ['Groundtruth X', 'Groundtruth Y', 'Groundtruth Z'],
          ['Distance (m)', 'Distance (m)', 'Time (s)'], binary_sight)

# %%

plot_data(attitude_state_time, attitude_px4_time,
          attitude_state_data, [1, 2, 3],
          attitude_px4_data, [3, 2, 1],
          ['Estimation Roll', 'Estimation Pitch', 'Estimation Yaw'],
          ['Groundtruth Roll', 'Groundtruth Pitch', 'Groundtruth Yaw'],
          ['Angle (degrees)', 'Angle (degrees)', 'Time (s)'])
plt.show()

# %%

# interpolate relative position data to match state estimation data timestamps
f_x = interpolate.interp1d(
    boat_time, relative_pos_gt[:, 0], kind='linear', fill_value="extrapolate")
f_y = interpolate.interp1d(
    boat_time, relative_pos_gt[:, 1], kind='linear', fill_value="extrapolate")
f_z = interpolate.interp1d(
    boat_time, relative_pos_gt[:, 2], kind='linear', fill_value="extrapolate")

relative_pos_gt_interp = np.zeros_like(state_data[:, 1:4])
relative_pos_gt_interp[:, 0] = f_x(state_time)
relative_pos_gt_interp[:, 1] = f_y(state_time)
relative_pos_gt_interp[:, 2] = f_z(state_time)

# calculate mean absolute error
MAE = np.mean(np.abs((state_data[:, 1:4] - relative_pos_gt_interp)), axis=0)
print('MAE: ', MAE)

plot_data(state_time, state_time,
          state_data, [1, 2, 3],
          relative_pos_gt_interp, [0, 1, 2],
          ['Estimation X', 'Estimation Y', 'Estimation Z'],
          ['Groundtruth X', 'Groundtruth Y', 'Groundtruth Z'],
          ['Distance (m)', 'Distance (m)', 'Time (s)'], binary_sight)
