# %%
import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
BAG_NAME = '18_0'

latest_state_file = sorted([f for f in os.listdir(
    SAVE_DIR) if f'{BAG_NAME}_state_data' in f])[-1]
print('latest_gt_file: ', latest_state_file)

# NpzFile 'gt.npz' with keys: drone_time, boat_time, drone_pos, boat_pos
gt_data = np.load(f'{SAVE_DIR}/{BAG_NAME}_gt.npz')
# load state estimation data from state_data.npy
state_data = np.load(f'{SAVE_DIR}/{latest_state_file}').reshape(-1, 14)

drone_time = gt_data['drone_time']
boat_time = gt_data['boat_time']
drone_pos = gt_data['drone_pos']
boat_pos = gt_data['boat_pos']

# cap boat data to match drone data
boat_data_start = np.argmin(np.abs(boat_time - drone_time[0]))
boat_data_end = np.argmin(np.abs(boat_time - drone_time[-1]))
boat_time = boat_time[boat_data_start:boat_data_end]
boat_pos = boat_pos[boat_data_start:boat_data_end, :]

# remove N last element that are too many in boat data
if boat_time.shape[0] > drone_time.shape[0]:
    N = boat_time.shape[0] - drone_time.shape[0]
    boat_time = boat_time[:-N]
    boat_pos = boat_pos[:-N, :]
else:
    N = drone_time.shape[0] - boat_time.shape[0]
    drone_time = drone_time[:-N]
    drone_pos = drone_pos[:-N, :]

state_time = state_data[:, 0]
target_in_sight = state_data[:, -1]
data_start = np.argmin(np.abs(drone_time - state_time[0]))
data_end = np.argmin(np.abs(drone_time - state_time[-1]))
drone_time = drone_time[data_start:data_end]
drone_pos = drone_pos[data_start:data_end, :]
boat_time = boat_time[data_start:data_end]
boat_pos = boat_pos[data_start:data_end, :]

print('drone_time.shape: ', drone_time.shape)
print('boat_time.shape: ', boat_time.shape)
print('drone_pos.shape: ', drone_pos.shape)
print('boat_pos.shape: ', boat_pos.shape)
print('state_data.shape: ', state_data.shape)

# %%

# plot the state estimation position data i.e. first 3 columns of state_data
relative_pos_gt = boat_pos - drone_pos
fig, axs = plt.subplots(3, 1, figsize=(10, 5), dpi=200)

binary_sight = np.where(target_in_sight > 0)

axs[0].scatter(state_time, state_data[:, 1], label='Estimation X', s=1, marker='*')
axs[0].scatter(drone_time, relative_pos_gt[:, 0], label='Groundtruth X', s=1)
axs[0].scatter(state_time[binary_sight],
               np.ones_like(state_time)[binary_sight],
               label='Target in FOV', s=1, color='green', marker='x')
axs[0].legend(markerscale=5, loc='lower right')
axs[0].grid(True, linestyle='-', linewidth=0.5)

# drone_y vs boat_y
axs[1].scatter(state_time, state_data[:, 2], label='Estimation Y', s=1, marker='*')
axs[1].scatter(drone_time, relative_pos_gt[:, 1], label='Groundtruth Y', s=1)
axs[1].scatter(state_time[binary_sight],
               np.ones_like(state_time)[binary_sight],
               label='Target in FOV', s=1, color='green', marker='x')
axs[1].legend(markerscale=5, loc='lower right')
axs[1].set_ylabel('Distance (m)')
axs[1].grid(True, linestyle='-', linewidth=0.5)

# drone_z vs boat_z
axs[2].scatter(state_time, state_data[:, 3], label='Estimation Z', s=1, marker='*')
axs[2].scatter(drone_time, relative_pos_gt[:, 2], label='Groundtruth Z', s=1)
axs[2].scatter(state_time[binary_sight],
               np.ones_like(state_time)[binary_sight],
               label='Target in FOV', s=1, color='green', marker='x')
axs[2].legend(markerscale=5, loc='lower right')
axs[2].set_xlabel('Time (s)')
axs[2].grid(True, linestyle='-', linewidth=0.5)

fig.align_xlabels()
fig.align_ylabels()
plt.tight_layout()
plt.show()
