#%%
import numpy as np
import matplotlib.pyplot as plt

BAG_NAME = '18_0'

# NpzFile 'gt.npz' with keys: drone_time, boat_time, drone_pos, boat_pos
gt_data = np.load(f'{BAG_NAME}_gt.npz')
drone_time = gt_data['drone_time']
boat_time = gt_data['boat_time']
drone_pos = gt_data['drone_pos']
boat_pos = gt_data['boat_pos']
# load state estimation data from state_data.npy
state_data = np.load(f'{BAG_NAME}_state_data.npy').reshape(-1, 13)

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
plt.figure(dpi=200)
plt.scatter(state_time, state_data[:, 1], label='drone_x', s=1)
plt.scatter(state_time, state_data[:, 2], label='drone_y', s=1)
plt.scatter(state_time, state_data[:, 3], label='drone_z', s=1)

relative_pos_gt =  drone_pos - boat_pos
plt.scatter(drone_time, -relative_pos_gt[:, 0], label='boat_x', s=1)
plt.scatter(drone_time, -relative_pos_gt[:, 1], label='boat_y', s=1)
plt.scatter(drone_time, -relative_pos_gt[:, 2], label='boat_z', s=1)
plt.legend(markerscale=5)
