# %%
import scipy.optimize as opt
import pymap3d as pm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# %%

# read cvs file record.csv
df = pd.read_csv('record6.csv')
df.columns, len(df)

# %%

# take the last row
# row = df.iloc[-1]
# remove this row
# df = df.iloc[:-1]

# lat, lng, alt = row[0], row[1], row[2]
lat, lng, alt = 55.602979999999995, 12.3868665, 1.0210000000000001

# %%

# find first row with not nan /gps_postproc/altitude column
i = 0
for i, a in enumerate(df["/gps_postproc/altitude"]):
    if not np.isnan(a):
        print(a, i)
        break

drone_pos_row = df.iloc[i]
# get index of column "/gps_postproc/lattitude"
dlat = drone_pos_row["/gps_postproc/latitude"]
dlng = drone_pos_row["/gps_postproc/longitude"]
dalt = drone_pos_row["/gps_postproc/altitude"]
dlat, dlng, dalt

# %%

pm.geodetic2enu(lat, lng, alt, dlat, dlng, dalt)

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
    if not np.isnan(row["/cam_target_pos/point/x"]):
        pos = (row["/cam_target_pos/point/x"],
               row["/cam_target_pos/point/y"],
               row["/cam_target_pos/point/z"])
        meas_pos.append(pos)
        meas_time.append(row["/cam_target_pos/header/stamp"])
    # if not np.isnan(row["/imu/data_world/header/stamp"]):
    #     acc.append([
    #         row["/imu/data_world/vector/x"],
    #         row["/imu/data_world/vector/y"],
    #         row["/imu/data_world/vector/z"]
    #     ])
    #     acc_time.append(row["/imu/data_world/header/stamp"])

# interpolate gt_pos to match meas_pos
gt_pos = np.array(gt_pos)
meas_pos = np.array(meas_pos)
acc = np.array(acc)
print(len(gt_pos), len(meas_pos), len(acc))

# %%

plt.scatter(gt_time, gt_pos[:, 1], label="X gt")
plt.scatter(meas_time, -meas_pos[:, 0], label="X meas")
plt.legend()
plt.figure()
plt.scatter(gt_time, gt_pos[:, 0], label="Y gt")
plt.scatter(meas_time, -meas_pos[:, 1], label="Y meas")
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

# %%


def get_2D_R(theta):
    # wrap it to between 0 and 2pi
    theta = theta % (2*np.pi)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def cost(theta, gt_pos, meas_pos):
    # theta will be vector for each measurement
    loss = 0.0
    for i in range(len(meas_pos)):
        R = get_2D_R(theta[i])
        loss += np.linalg.norm((R @ meas_pos[i].T).T - gt_pos[i], ord=2)
    return loss


corresponding_pairs = []
for k, gtt in enumerate(gt_time):
    for j, meast in enumerate(meas_time):
        if j == 0:
            continue
        if j == len(meas_time)-1:
            break
        diff = abs(gtt - meast)
        next_diff = abs(gtt - meas_time[j+1])
        prev_diff = abs(gtt - meas_time[j-1])
        if diff < 0.01 and next_diff > diff and prev_diff > diff:
            corresponding_pairs.append((k, j))
            break

print(len(corresponding_pairs))

X = np.array([gt_pos[i, :2] for i, _ in corresponding_pairs])
Y = np.array([meas_pos[j, :2] for _, j in corresponding_pairs])
times = np.array([gt_time[i] for i, _ in corresponding_pairs])

bounds = [(0, 2*np.pi) for _ in range(len(corresponding_pairs))]

opt_res = opt.minimize(cost, np.ones((len(corresponding_pairs))) * np.pi, args=(X, Y), bounds=bounds)
print(opt_res)
assert opt_res.success

#%%
opt_res.x

# %%

# rotate the each measurement correcpondance by the found theta
# R = get_2D_R(opt_res.x)
new_meas = Y.copy()
for i in range(len(new_meas)):
    theta = opt_res.x[i]
    R = get_2D_R(theta)
    new_meas[i, :] = (R @ new_meas[i, :].T).T

# plot X vs new_meas
plt.scatter(times, new_meas[:, 0], label="X new_meas")
plt.scatter(times, X[:, 0], label="X gt", linestyle="--")
plt.legend()
plt.figure()
plt.scatter(times, new_meas[:, 1], label="Y new_meas")
plt.scatter(times, X[:, 1], label="Y gt", linestyle="--")
# plot norms
plt.figure()
plt.scatter(times, np.linalg.norm(new_meas, axis=1), label="new norm")
plt.scatter(times, np.linalg.norm(X, axis=1), label="gt norm", linestyle="--")
plt.legend()


# %%

# plot x,y,z on seperate plots comparing ground truth and measured
# they are not the same length
plt.figure()
plt.scatter(gt_time, gt_pos[:, 0], label="X gt")
plt.scatter(meas_time, new_meas[:, 0], label="X meas")
plt.legend()
plt.figure()
plt.scatter(gt_time, gt_pos[:, 1], label="Y gt")
plt.scatter(meas_time, new_meas[:, 1], label="Y meas")
plt.legend()
plt.figure()
plt.scatter(gt_time, gt_pos[:, 2], label="Z gt")
plt.scatter(meas_time, new_meas[:, 2], label="Z meas")
plt.legend()

# plot norms
plt.figure()
plt.plot(gt_time, np.linalg.norm(gt_pos, axis=1), label="gt")
plt.plot(meas_time, np.linalg.norm(new_meas, axis=1), label="meas")
plt.legend()
plt.show()

# %%

#  - [ ] So the orientation has some bias that cannot be determined; for filter tunning do this:
# 	 - [ ] Take the magnitude from ground truth
# 	 - [ ] Direction (unit vector) from the measurement
# 	 - [ ] Scale the vector with the ground truth magnitude
# 	 - [ ] This will bring error norm to ~zero

gt_corr = np.array([gt_pos[i, :] for i, _ in corresponding_pairs])
meas_corr = np.array([meas_pos[j, :] for _, j in corresponding_pairs])
times = np.array([gt_time[i] for i, _ in corresponding_pairs])

corresponding_pairs = []
for k, gtt in enumerate(gt_time):
    for j, meast in enumerate(meas_time):
        if j == 0:
            continue
        if j == len(meas_time)-1:
            break
        diff = abs(gtt - meast)
        next_diff = abs(gtt - meas_time[j+1])
        prev_diff = abs(gtt - meas_time[j-1])
        if diff < 0.1 and next_diff > diff and prev_diff > diff:
            corresponding_pairs.append((k, j))
            break
print(corresponding_pairs)
# check if all pairs are unique and no duplicates indexes

# %%

# magnitude from ground truth
mag = np.linalg.norm(gt_corr, axis=1)
# direction from measurement
dir = meas_corr / np.linalg.norm(meas_corr, axis=1)[:, None]
# scale the vector with the ground truth magnitude
new_meas = dir * mag[:, None]
plt.scatter(times, new_meas[:, 0], label="new meas")
plt.scatter(times, meas_corr[:, 0], label="prev", linestyle="--")
plt.legend()
plt.figure()
plt.scatter(times, new_meas[:, 1], label="new meas")
plt.scatter(times, meas_corr[:, 1], label="prev", linestyle="--")
plt.legend()
plt.figure()
plt.scatter(times, new_meas[:, 2], label="new meas")
plt.scatter(times, meas_corr[:, 2], label="prev", linestyle="--")
plt.legend()
plt.figure()
plt.scatter(times, np.linalg.norm(new_meas, axis=1), label="new norm")
plt.scatter(gt_time, np.linalg.norm(gt_pos, axis=1),
            label="gt norm", linestyle="--")
plt.legend()
plt.show()


# %%

# acc.shape, len(acc_time)
meas_time = np.array(meas_time)

# interpolate meas time to match acc time
interpolated_meas_time = np.interp(acc_time, meas_time, meas_time)
interpolated_meas_time.shape, len(acc_time)

# interpolated_meas = np.interp(acc_time, times, new_meas[:,0])
# plt.plot(interpolated_meas)

interpolated_gt = np.interp(acc_time, gt_time, gt_pos[:, 0])
plt.plot(interpolated_gt)


# i need measurements of P and measurements of acc that's already in place
# groundtruth of state which is position, velocity and acceleration : (
