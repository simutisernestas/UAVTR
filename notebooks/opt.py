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

opt_res = opt.minimize(cost, np.ones(
    (len(corresponding_pairs))) * np.pi, args=(X, Y), bounds=bounds)
print(opt_res)
assert opt_res.success

# %%
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