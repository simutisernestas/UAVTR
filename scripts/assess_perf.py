# %%
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

COLOR1 = 'royalblue'
COLOR2 = 'firebrick'
COLOR3 = 'darkorange'

# time state target_in_sight cov_p cov_v
# 1 + 5 * 3 + 1 + 3 + 3 = 23
STATE_RECORD_SIZE = 17 + 3 + 3 + 2
STATE_TIME_COLUMN = 0
STATE_TARGET_IN_SIGHT_COLUMN = 13 + 3
STATE_COV_X_COLUMN = 14 + 3
STATE_COV_Y_COLUMN = 15 + 3
STATE_COV_Z_COLUMN = 16 + 3
VEL_COV_X_COLUMN = 17 + 3
VEL_COV_Y_COLUMN = 18 + 3
VEL_COV_Z_COLUMN = 19 + 3
STATE_DRONE_VEL_X_COLUMN = 4
STATE_DRONE_VEL_Y_COLUMN = 5
STATE_DRONE_VEL_Z_COLUMN = 6
SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
BAGS_LIST = [
    '18_0',
    'latest_flight_mode0',
    'latest_flight_mode1',
    'latest_flight_mode2',
]
CUTOFF_TIMES = [-35, -45, 0, -85]
NTH_FROM_BACK = 1
LIVE = len(sys.argv) == 1 or not sys.argv[1].isdigit()
PLOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/plots'
os.makedirs(PLOT_DIR, exist_ok=True)
WHICH = int(sys.argv[1]) if not LIVE else 1
BAG_NAME = BAGS_LIST[WHICH]

def load_latest(name, shape):
    latest_file = sorted([f for f in os.listdir(
        SAVE_DIR) if f'{BAG_NAME}_{name}' in f])[-NTH_FROM_BACK]
    print(f'latest {name} file: ', latest_file)
    return np.load(f'{SAVE_DIR}/{latest_file}').reshape(-1, shape)


vel_measurements_data = load_latest("vel_measurements", 4)
vel_measurements_time = vel_measurements_data[:, 0]

latest_state_file = sorted([f for f in os.listdir(
    SAVE_DIR) if f'{BAG_NAME}_state_data' in f])[-NTH_FROM_BACK]
print('latest state file: ', latest_state_file)
state_timestamp = latest_state_file.split('_')[0]
# NpzFile 'gt.npz' with keys: drone_time, boat_time, drone_pos, boat_pos
GT_NAME = BAG_NAME if "mode" not in BAG_NAME else "_".join(
    BAG_NAME.split('_')[:-1])
gt_data = np.load(f'{SAVE_DIR}/{GT_NAME}_gt.npz')
# load state estimation data from state_data.npy
state_data = np.load(
    f'{SAVE_DIR}/{latest_state_file}').reshape(-1, STATE_RECORD_SIZE)

# load attitude estimation data from timstamp_bag_attitude_state.npy
latest_attitude_state_file = sorted([f for f in os.listdir(
    SAVE_DIR) if f'{BAG_NAME}_attitude_state' in f])[-NTH_FROM_BACK]
attitude_state_data = np.load(
    f'{SAVE_DIR}/{latest_attitude_state_file}').reshape(-1, 4)
print('attitude_state_data.shape: ', attitude_state_data.shape)
attitude_state_time = attitude_state_data[:, 0]

# load attitude px4 data from timstamp_bag_attitude_px4.npy
latest_attitude_px4_file = sorted([f for f in os.listdir(
    SAVE_DIR) if f'{BAG_NAME}_attitude_px4' in f])[-NTH_FROM_BACK]
attitude_px4_data = np.load(
    f'{SAVE_DIR}/{latest_attitude_px4_file}').reshape(-1, 4)
print('attitude_px4_data.shape: ', attitude_px4_data.shape)
attitude_px4_time = attitude_px4_data[:, 0]

drone_time = gt_data['drone_time']
boat_time = gt_data['boat_time']
drone_pos = gt_data['drone_pos']
boat_pos = gt_data['boat_pos']
drone_vel = gt_data['drone_vel']
boat_vel = gt_data['boat_vel']

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
    state_time -= 104
    attitude_state_time -= 104
    attitude_px4_time -= 104

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
boat_vel = boat_vel[matching_indices[1], :]

print('drone_time.shape: ', drone_time.shape)
print('boat_time.shape: ', boat_time.shape)
print('drone_pos.shape: ', drone_pos.shape)
print('boat_pos.shape: ', boat_pos.shape)
print('state_data.shape: ', state_data.shape)
print('drone_vel.shape: ', drone_vel.shape)

if BAG_NAME == BAGS_LIST[0]:
    t0 = np.argmin(np.abs(500 - boat_time))
    boat_pos_var = np.var(boat_pos[t0:, :], axis=0)
    boat_pos_std = np.sqrt(boat_pos_var)
    groundtruth_3std = boat_pos_std*3*2
    print("3 std: ", groundtruth_3std)
    static_boat_pos = np.mean(boat_pos[t0:, :], axis=0)
    relative_pos_gt = static_boat_pos - drone_pos[:boat_pos.shape[0], :]
elif BAG_NAME == BAGS_LIST[1]:
    time = 1697639181.199643 + 200
    t0 = np.argmin(np.abs(time - boat_time))
    t1 = np.argmin(np.abs(time + 100 - boat_time))
    boat_pos_var = np.var(boat_pos[t0:t1, :], axis=0)
    boat_pos_std = np.sqrt(boat_pos_var)
    groundtruth_3std = boat_pos_std*3*2
    print("3 std: ", groundtruth_3std)
    # static_boat_pos = np.mean(boat_pos[t0:t1, :], axis=0)
    relative_pos_gt = boat_pos - drone_pos
else:
    # time = 1697639391.400244 + 200
    # t0 = np.argmin(np.abs(time - boat_time))
    # t1 = np.argmin(np.abs(time + 100 - boat_time))
    # boat_pos_var = np.var(boat_pos[t0:t1, :], axis=0)
    # boat_pos_std = np.sqrt(boat_pos_var)
    # groundtruth_3std = boat_pos_std*3*2
    groundtruth_3std = np.array([2.9305771, 2.84174148, 1.16107856])
    print("3 std: ", groundtruth_3std)
    relative_pos_gt = boat_pos - drone_pos

# binary signal indicating whether the target is in sight or not
target_in_sight = state_data[:, STATE_TARGET_IN_SIGHT_COLUMN]
binary_sight = target_in_sight > 0

# %%


def plot_data(t0_data, t1_data, state_data, state_index, pos_data, pos_index, est_label, gt_label, axis_label, bns=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), dpi=200)
    for i, (state_idx, pos_idx, est_lbl, gt_lbl, axis_lbl) in enumerate(zip(state_index, pos_index, est_label, gt_label, axis_label)):
        axs[i].scatter(t0_data, state_data[:, state_idx],
                       label=est_lbl, s=1, marker='*', c=COLOR1)
        axs[i].scatter(t1_data, pos_data[:, pos_idx],
                       label=gt_lbl, s=1, c=COLOR2)

        pos_index_at_t00 = np.argmin(np.abs(t0_data[0] - t1_data))
        pos_index_at_t01 = np.argmin(
            np.abs(t0_data[-1] + CUTOFF_TIMES[WHICH] - t1_data))
        if WHICH == 10:
            min_y = np.min(
                pos_data[pos_index_at_t00:pos_index_at_t01, pos_idx])
            max_y = np.max(
                pos_data[pos_index_at_t00:pos_index_at_t01, pos_idx])
        else:
            min_y = np.min([np.min(state_data[:, state_idx]), np.min(
                pos_data[pos_index_at_t00:pos_index_at_t01, pos_idx])])
            max_y = np.max([np.max(state_data[:, state_idx]),
                            np.max(pos_data[pos_index_at_t00:pos_index_at_t01, pos_idx])])
        axs[i].set_ylim([min_y - 5.0, max_y + 5.0])
        axs[i].set_xlim([t0_data[0], t0_data[-1] + CUTOFF_TIMES[WHICH]])

        if bns is not None:
            axs[i].scatter(t0_data[bns], np.ones_like(t0_data)[
                           bns] * max_y + 3.0, label='Target in FOV', s=1, color='green', marker='x')
            std = 3*np.sqrt(state_data[:, state_idx + STATE_COV_X_COLUMN - 1])
            axs[i].fill_between(t0_data, state_data[:, state_idx] -
                                std, state_data[:, state_idx] + std,
                                alpha=0.1, color=COLOR1)
            axs[i].fill_between(
                t1_data, pos_data[:, pos_idx] - groundtruth_3std[pos_idx],
                pos_data[:, pos_idx] + groundtruth_3std[pos_idx],
                alpha=0.1, color=COLOR2)

        axs[i].legend(markerscale=5)
        axs[i].grid(True, linestyle='-', linewidth=0.5)
        if i == 1:
            axs[i].set_ylabel(axis_lbl)
        elif i == 2:
            axs[i].set_xlabel(axis_lbl)

    fig.align_xlabels()
    fig.align_ylabels()
    fig.tight_layout()
    return fig

# %%


plot_data(state_time, drone_time,
          state_data, [1, 2, 3],
          relative_pos_gt, [0, 1, 2],
          ['Estimation X', 'Estimation Y', 'Estimation Z'],
          ['Groundtruth X', 'Groundtruth Y', 'Groundtruth Z'],
          ['Distance (m)', 'Distance (m)', 'Time (s)'], binary_sight)
if LIVE:
    plt.show()
else:
    plt.savefig(f'{PLOT_DIR}/{state_timestamp}_{BAG_NAME}_state.png')
# %%

plot_data(attitude_state_time, attitude_px4_time,
          attitude_state_data, [1, 2, 3],
          attitude_px4_data, [3, 2, 1],
          ['Estimation Roll', 'Estimation Pitch', 'Estimation Yaw'],
          ['Groundtruth Roll', 'Groundtruth Pitch', 'Groundtruth Yaw'],
          ['Angle (degrees)', 'Angle (degrees)', 'Time (s)'])
if LIVE:
    plt.show()
else:
    plt.savefig(f'{PLOT_DIR}/{state_timestamp}_{BAG_NAME}_attitude.png')

# %%

# f = interpolate.interp1d(
#     attitude_px4_time, attitude_px4_data[:, 1], kind='linear', fill_value="extrapolate")
# px4_yaw_interp = f(attitude_state_time)
# yaw_diff = np.abs(px4_yaw_interp - attitude_state_data[:, 3])
# yaw_diff_idx = np.argmax(yaw_diff < 5)

# # interpolate relative position data to match state estimation data timestamps
# f_x = interpolate.interp1d(
#     boat_time, relative_pos_gt[:, 0], kind='linear', fill_value="extrapolate")
# f_y = interpolate.interp1d(
#     boat_time, relative_pos_gt[:, 1], kind='linear', fill_value="extrapolate")
# f_z = interpolate.interp1d(
#     boat_time, relative_pos_gt[:, 2], kind='linear', fill_value="extrapolate")

# data_frac = state_data[yaw_diff_idx:, 1:4]
# time_frac = state_time[yaw_diff_idx:]

# relative_pos_gt_interp = np.zeros_like(data_frac)
# relative_pos_gt_interp[:, 0] = f_x(time_frac)
# relative_pos_gt_interp[:, 1] = f_y(time_frac)
# relative_pos_gt_interp[:, 2] = f_z(time_frac)

# # calculate mean absolute error
# MAE = np.mean(
#     np.abs((data_frac - relative_pos_gt_interp[:])), axis=0)
# print('MAE: ', MAE)

# fig = plot_data(time_frac, time_frac,
#                 state_data[yaw_diff_idx:, :], [1, 2, 3],
#                 relative_pos_gt_interp, [0, 1, 2],
#                 ['Estimation X', 'Estimation Y', 'Estimation Z'],
#                 ['Groundtruth X', 'Groundtruth Y', 'Groundtruth Z'],
#                 ['Distance (m)', 'Distance (m)', 'Time (s)'],
#                 bns=binary_sight[yaw_diff_idx:])

# if LIVE:
#     plt.show()
# else:
#     plt.savefig(f'{PLOT_DIR}/{state_timestamp}_{BAG_NAME}_interp_mae.png')

# %%

# create subfigure for each axis
fig, axs = plt.subplots(3, 1, figsize=(10, 7), dpi=200)
state_vel_order = [STATE_DRONE_VEL_Y_COLUMN,
                   STATE_DRONE_VEL_X_COLUMN, STATE_DRONE_VEL_Z_COLUMN]
state_vel_cov_order = [VEL_COV_Y_COLUMN,
                       VEL_COV_X_COLUMN, VEL_COV_Z_COLUMN]
state_labels = ['Estimation X', 'Estimation Y', 'Estimation Z']
gt_labels = ['Groundtruth X', 'Groundtruth Y', 'Groundtruth Z']
axis_lbl_x = 'Time (s)'
axis_lbl_y = 'Velocity (m/s)'
for i in range(3):
    mult = -1 if i == 2 else 1
    axs[i].scatter(drone_time, mult * drone_vel[:, i],
                   label=gt_labels[i], s=1, c=COLOR1)
    axs[i].scatter(state_time, state_data[:, state_vel_order[i]],
                   label=state_labels[i], s=1, c=COLOR2)

    std = 3*np.sqrt(state_data[:, state_vel_cov_order[i]])

    axs[i].fill_between(state_time,
                        state_data[:, state_vel_order[i]] - std,
                        state_data[:, state_vel_order[i]] + std,
                        alpha=0.1, color=COLOR2)

    axs[i].set_xlim([state_time[0], state_time[-1] + CUTOFF_TIMES[WHICH]])
    offset = 1
    y_min = np.min(mult * drone_vel[:, i]) - offset
    y_max = np.max(mult * drone_vel[:, i]) + offset
    axs[i].set_ylim([y_min, y_max])

    axs[i].legend(markerscale=5)
    axs[i].grid(True, linestyle='-', linewidth=0.5)
    if i == 1:
        axs[i].set_ylabel(axis_lbl_y)
    elif i == 2:
        axs[i].set_xlabel(axis_lbl_x)

fig.align_xlabels()
fig.align_ylabels()
fig.tight_layout()
if LIVE:
    plt.show()
else:
    plt.savefig(f'{PLOT_DIR}/{state_timestamp}_{BAG_NAME}_drone_vel.png')

# %%

# boat velocity estimation
fig, axs = plt.subplots(2, 1, figsize=(10, 7), dpi=200)
state_vel_order = [STATE_DRONE_VEL_Y_COLUMN + 3,
                   STATE_DRONE_VEL_X_COLUMN + 3]
state_vel_cov_order = [-2, -1]
state_labels = ['Estimation X', 'Estimation Y', 'Estimation Z']
gt_labels = ['Groundtruth X', 'Groundtruth Y', 'Groundtruth Z']
axis_lbl_x = 'Time (s)'
axis_lbl_y = 'Velocity (m/s)'
boat_time -= 70
for i in range(2):
    mult = -1 if i == 2 else 1
    axs[i].scatter(boat_time, boat_vel[:, i],
                   label=gt_labels[i], s=1, c=COLOR1)
    axs[i].scatter(state_time, state_data[:, state_vel_order[i]],
                   label=state_labels[i], s=1, c=COLOR2)
    axs[i].set_xlim([state_time[0], state_time[-1] + CUTOFF_TIMES[WHICH]])

    std = 3*np.sqrt(state_data[:, state_vel_cov_order[i]])
    axs[i].fill_between(state_time,
                        state_data[:, state_vel_order[i]] - std,
                        state_data[:, state_vel_order[i]] + std,
                        alpha=0.1, color=COLOR2)

    if WHICH == 0:
        axs[i].set_ylim([-1, 1])
    else:
        offset = .3
        y_min = np.min(boat_vel[:, i]) - offset
        y_max = np.max(boat_vel[:, i]) + offset
        axs[i].set_ylim([y_min, y_max])

    axs[i].legend(markerscale=5)
    axs[i].grid(True, linestyle='-', linewidth=0.5)
    axs[i].set_ylabel(axis_lbl_y)
    if i == 1:
        axs[i].set_xlabel(axis_lbl_x)

fig.align_xlabels()
fig.align_ylabels()
fig.tight_layout()
if LIVE:
    plt.show()
else:
    plt.savefig(f'{PLOT_DIR}/{state_timestamp}_{BAG_NAME}_boat_vel.png')

# %%
