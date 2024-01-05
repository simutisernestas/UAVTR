# %%
import os
import cv2
import spatialmath as sm
import spatialmath.base as smb
import transforms3d as tf
import scipy.optimize as opt
import scipy
import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
# nice print of numpy array
np.set_printoptions(precision=7, suppress=True)

# compute variance boat pos on each dimension
boat_pos  # Nx3 vector

t0 = np.argmin(np.abs(500 - boat_time))
boat_pos_var = np.var(boat_pos[t0:, :], axis=0)
boat_pos_std = np.sqrt(boat_pos_var)
groundtruth_3std = boat_pos_std*3
static_boat_pos = np.mean(boat_pos[t0:, :], axis=0)

# add MAE to the plot
fig.text(0.5, 0.95, f'MAE: {MAE}', horizontalalignment='center',
         verticalalignment='center', wrap=True, fontsize=12)

# # plot norm of both state position and relative ground truth position data
# state_pos_norm = np.linalg.norm(state_data[:, 1:4], axis=1)
# relative_pos_gt_norm = np.linalg.norm(relative_pos_gt, axis=1)
# plt.plot(state_time, state_pos_norm, label='Estimation')
# plt.plot(drone_time, relative_pos_gt_norm, label='Groundtruth')
# plt.legend()
# plt.grid(True, linestyle='-', linewidth=0.5)
# plt.tight_layout()
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (m)')
# if LIVE:
#     plt.show()


# plot boat pos
plt.figure()
plt.plot(boat_time[:], boat_pos[:, 0], label='X')
plt.plot(boat_time[:], boat_pos[:, 1], label='Y')
plt.plot(boat_time[:], boat_pos[:, 2], label='Z')
# plt.plot(boat_time[t0:t1], boat_pos[t0:t1, 0], label='X')
# plt.plot(boat_time[t0:t1], boat_pos[t0:t1, 1], label='Y')
# plt.plot(boat_time[t0:t1], boat_pos[t0:t1, 2], label='Z')
# fill between with groundtruth_3std from pos_data
# plt.fill_between(
#     boat_time[t0:t1], boat_pos[t0:t1, 0] - groundtruth_3std[0],
#     boat_pos[t0:t1, 0] + groundtruth_3std[0],
#     alpha=0.1, color='red')
# plt.fill_between(
#     boat_time[t0:t1], boat_pos[t0:t1, 1] - groundtruth_3std[1],
#     boat_pos[t0:t1, 1] + groundtruth_3std[1],
#     alpha=0.1, color='red')
# plt.fill_between(
#     boat_time[t0:t1], boat_pos[t0:t1, 2] - groundtruth_3std[2],
#     boat_pos[t0:t1, 2] + groundtruth_3std[2],
#     alpha=0.1, color='red')
plt.legend()
plt.grid(True, linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')

# # same for drone pos
# drone_pos_var = np.var(drone_pos, axis=0)

# # plot drone pos
# plt.figure()
# plt.plot(drone_time, drone_pos[:, 0], label='X')
# plt.plot(drone_time, drone_pos[:, 1], label='Y')
# plt.plot(drone_time, drone_pos[:, 2], label='Z')
# plt.legend()
# plt.grid(True, linestyle='-', linewidth=0.5)
# plt.tight_layout()
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (m)')

# state_time


def e2h(e):
    if e.ndim == 1:
        return np.append(e, 1)

    return np.vstack((e, np.ones(e.shape[1])))


def h2e(h):
    if h.ndim == 1:
        return h[:-1]/h[-1]

    return h[:-1, :]/h[-1, :]


def project(P, C, noise=0, normalize=True):
    e = np.random.normal(0, noise, (3, P.shape[1]))
    projected = C @ e2h(P)
    if not normalize:
        return projected + e
    return h2e(C @ e2h(P)) + e[:2, :]


def Lx(p_xy, Zs):

    Lx = np.zeros((p_xy.shape[0] * 2, 6))

    for i in range(p_xy.shape[0]):
        x = p_xy[i, 0]
        y = p_xy[i, 1]
        Z = Zs[i]

        Lx[2*i:2*i+2, :] = np.array([
            [-1/Z,  0,     x/Z, x * y,      -(1 + x**2), y,],
            [0,   -1/Z,   y/Z, (1 + y**2), -x*y,       -x]])

    return Lx


def plot_scene(rot_matrix, origin):
    fig = plt.figure()

    # Create 3D axes
    ax = fig.add_subplot(111, projection='3d')

    # Create unit vectors
    i = np.array([5, 0, 0])
    j = np.array([0, 5, 0])
    k = np.array([0, 0, 5])

    # Transform unit vectors
    i_transformed = np.dot(rot_matrix, i)
    j_transformed = np.dot(rot_matrix, j)
    k_transformed = np.dot(rot_matrix, k)

    # Plot transformed unit vectors
    ax.quiver(origin[0], origin[1], origin[2], i_transformed[0],
              i_transformed[1], i_transformed[2], color='r')
    ax.quiver(origin[0], origin[1], origin[2], j_transformed[0],
              j_transformed[1], j_transformed[2], color='g')
    ax.quiver(origin[0], origin[1], origin[2], k_transformed[0],
              k_transformed[1], k_transformed[2], color='b')

    # add origin text
    ax.text(origin[0], origin[1], origin[2], str(origin))

    # plot points in 3D
    ax.scatter(Points[:, 0], Points[:, 1], Points[:, 2], c='r', s=.1, alpha=.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # # Set limits and labels
    ax.set_xlim([-lower_bound, lower_bound])
    ax.set_ylim([-1, lower_bound+lower_bound])
    ax.set_zlim([-lower_bound-Z_var, 3])

    ax.set_aspect('equal')

    plt.show()

# %%


DEG2RAD = np.pi/180
u0 = 640//2    # principal point, horizontal coordinate
v0 = 480//2    # principal point, vertical coordinate
K = np.array([[285, 0, u0],
              [0, 285, v0],
              [0, 0, 1]])
Kinv = np.linalg.inv(K)
P0 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=float)
T0 = sm.SE3() \
    * sm.SE3.Rx((90+45)*DEG2RAD)
R0 = T0.inv().R
P0 = T0.A[:3, :]
dt = 1/30
dx = np.random.uniform(0, 2) * dt
dy = np.random.uniform(0, 2) * dt
dz = np.random.uniform(0, 2) * dt

T = T0 \
    * sm.SE3.Rz(np.random.uniform(-1, 1) * DEG2RAD) \
    * sm.SE3.Ry(np.random.uniform(-1, 1) * DEG2RAD) \
    * sm.SE3.Rx(np.random.uniform(-1, 1) * DEG2RAD) \
    * sm.SE3.Tx(dx) \
    * sm.SE3.Ty(dy) \
    * sm.SE3.Tz(dz)  # relative between cameras
print(f"delta rel: {smb.tr2delta(T0.A, T.A)/dt}")
R1 = T.inv().R
P1 = T.A[:3, :]
C0 = K @ P0  # @ X
C1 = K @ P1  # @ X

gt = P1[:, 3] / dt
gt = T.inv().delta(T0.inv())/dt
ANGVEL = gt[3:]

# %%


def simulation(lower_bound, plot=False, depth_assumption=False):
    NOISE = 10 / lower_bound * 0
    # Vectorize projection calculations
    projected0 = project(Points.T, C0, NOISE)
    projected0_xy = Kinv @ e2h(projected0)
    projected1 = project(Points.T, C1, NOISE)
    projected1_xy = Kinv @ e2h(projected1)

    # Vectorize flow calculations
    flows = projected1 - projected0
    flows_xy = (projected0_xy[:2, :] - projected1_xy[:2, :]).T
    # flows_xy += np.random.normal(0, (np.abs(flows_xy)).max()/1000, flows_xy.shape)

    # Depth assumption
    if depth_assumption:
        Points[:, 2] = lower_bound
    # compute depths
    # Zs = np.linalg.norm(P1[:3, 3] - (R1 @ Points.T).T, axis=1)
    Zs = np.linalg.norm(P1[:3, 3] - Points, axis=1)
    # Zs = Points[:, 2] - dz
    # print(Zs.min(), Zs.max(), Zs.mean(), Zs.std())
    # Zs = np.linalg.norm(Points, axis=1)

    Lx1 = Lx(projected1_xy.T, Zs)
    Lx1 = np.vstack(Lx1)
    invL = np.linalg.pinv(Lx1)

    flows_xy = flows_xy.reshape(-1, 1) / dt

    # Plotting remains the same
    if plot:
        plt.figure(dpi=300)
        plt.scatter(projected0[0, :], projected0[1, :], c='r', s=.1)
        plt.scatter(projected1[0, :], projected1[1, :], c='g', s=.1)
        plt.quiver(projected0[0, :], projected0[1, :], flows[0, :], flows[1, :],
                   color='b', label='flow', angles='xy', scale_units='xy', scale=1)
        plt.xlim(0, u0 * 2)
        plt.ylim(0, v0 * 2)
        plt.gca().invert_yaxis()
        plt.show()
        # print(invL, Points)

        ns0 = scipy.linalg.null_space(Lx1[:2, :])
        ns1 = scipy.linalg.null_space(Lx1[2:4, :])
        ns2 = scipy.linalg.null_space(Lx1[4:, :])
        for ns in [ns0, ns1, ns2]:
            print(ns)
            assert ns.shape[0] == 6, ns.shape
            assert ns.shape[1] == 4, ns.shape
        print(ns0.max(axis=0), ns0.min(axis=0))
        print(ns1.max(axis=0), ns1.min(axis=0))
        print(ns2.max(axis=0), ns2.min(axis=0))
        import itertools
        # for all combinations of null spaces
        for comb in itertools.combinations([ns0, ns1, ns2], 2):
            c_dist = np.linalg.norm(comb[0] - comb[1])
            print(c_dist)

    # good = []
    # for k in range(Points.shape[0]):
    #     ns = scipy.linalg.null_space(Lx1[k*2:k*2+2, :])
    #     if (np.abs(ns).max(axis=0) < .9)[1:].all():
    #         good.append(k)
    #         print(k)
    #     # if len(good) == 3:
    #     #     break

    # if len(good) < 3:
    #     raise Exception("not enough good points")

    # Lx1 = Lx1[[2*k for k in good] + [2*k+1 for k in good], :]
    # flows_xy = flows_xy[[2*k for k in good] + [2*k+1 for k in good], :]
    # invL = np.linalg.pinv(Lx1)

    # # subtract angular velocity
    # flows_xy -= (Lx1[:, 3:] @ ANGVEL).reshape(-1, 1)
    # v = invL[:3, :] @ flows_xy
    # return v

    v = invL @ flows_xy
    return v


for _ in range(1000):
    lower_bound = 20
    Z_var = 3
    # Existing code for Points initialization
    Points = np.random.uniform(-lower_bound, lower_bound, (4000, 3))
    Points[:, 1] += lower_bound*1.5
    Points[:, 2] = -np.random.uniform(
        lower_bound, lower_bound+Z_var, Points.shape[0])

    est = simulation(lower_bound, depth_assumption=False, plot=False)
    gtp = gt.reshape(-1, 1)
    # est, gt.reshape(-1, 1), np.linalg.norm(est[:3] - gtp[:3], ord=1)
    error = np.linalg.norm(est[:3] - gtp[:3], ord=1)
    print(f"error: {error}")
    if np.linalg.norm(est[:3] - gtp[:3], ord=1) < 0.1:
        simulation(lower_bound, depth_assumption=False, plot=True)
        print("found", error)
        break
    break

# %%


plot_scene(R0, np.array([0, 0, 0]))
plot_scene(R1, np.array([dx, dy, dz]))


# %%

NOISE = 20 / lower_bound * 0
# Vectorize projection calculations
projected0 = project(Points.T, C0, NOISE)
projected0_xy = Kinv @ e2h(projected0)
projected1 = project(Points.T, C1, NOISE)
projected1_xy = Kinv @ e2h(projected1)

flows = projected1 - projected0
flows_xy = (projected0_xy[:2, :] - projected1_xy[:2, :]).T

# Plotting remains the same
plt.figure(dpi=300)
plt.scatter(projected0[0, :], projected0[1, :], c='r', s=.1)
plt.scatter(projected1[0, :], projected1[1, :], c='g', s=.1)
plt.quiver(projected0[0, :], projected0[1, :], flows[0, :], flows[1, :],
           color='b', label='flow', angles='xy', scale_units='xy', scale=1)
plt.xlim(0, u0 * 2)
plt.ylim(0, v0 * 2)
plt.gca().invert_yaxis()
plt.show()

# %%

pt = np.array([[0, 0, -5]]).reshape(3, 1)
Z0 = la.inv(R0) @ pt
projected0 = project(pt, C0, 0)
projected0_xy = Kinv @ e2h(projected0)
repr0 = projected0_xy * Z0
projected1 = project(pt, C1, 0)
projected1_xy = Kinv @ e2h(projected1)
d = np.linalg.norm((P1[:3, 3] - pt.reshape(3)))
Z1 = (la.inv(R1) @ pt + P1[:3, 3].reshape(3, 1))[2]
repr1 = projected1_xy * Z1
# convert back to world
P1h = la.inv(np.concatenate(
    (P1, np.array([[0, 0, 0, 1]]).reshape(1, 4)), axis=0))
repr1 = e2h(repr1)
print((P1h @ repr1)[:3].T)
print((R0 @ repr0).T)
assert np.allclose((P1h @ repr1)[:3], R0 @ repr0)

# %%


def calculate_projection(pt, C, P, Kinv, R, show=False):
    projected = project(pt, C, 0)
    projected_xy = Kinv @ e2h(projected)
    d = np.linalg.norm((P[:3, 3] - pt.reshape(3)))
    pt_c = projected_xy * d
    pt_e = d * R @ projected_xy
    if show:
        print(projected)
        print(projected_xy)
        print(P[:3, 3])
        print(d)
        print(pt_c)
        print(pt_e)
    return projected, projected_xy, d, pt_c, pt_e


# Usage:
pt = np.array([[1, 5, -5]]).reshape(3, 1)
projected0, projected0_xy, d0, pt0_c, pt0_e = calculate_projection(
    pt, C0, P0, Kinv, R0, show=True)
print("-----")
projected1, projected1_xy, d1, pt1_c, pt1_e = calculate_projection(
    pt, C1, P1, Kinv, R1, show=True)

# np.linalg.norm((pt1_e - pt0_e)/dt, ord=1), vel
((pt1_e - pt0_e)/dt).reshape(-1), gt[:3]

# %%

# def skew_symmetric_matrix(v):
#     # v = [-x for x in v]
#     return np.array([[0, -v[2], v[1]],
#                      [v[2], 0, -v[0]],
#                      [-v[1], v[0], 0]])


# N = 5000
# lower_bound = 20
# in_z = 3
# # Existing code for Points initialization
# Points = np.random.uniform(-lower_bound/4, lower_bound/4, (N, 3))
# Points[:, 2] = np.random.uniform(
#     lower_bound, lower_bound+in_z, Points.shape[0])

# NOISE = 10 / lower_bound
# projected0 = project(Points.T, C0, NOISE)
# projected0_xy = Kinv @ e2h(projected0)
# projected1 = project(Points.T, C1, NOISE)
# projected1_xy = Kinv @ e2h(projected1)
# ang_comp = skew_symmetric_matrix(ANGVEL)*dt + np.eye(3)
# Rcw = np.linalg.inv(T.R)
# assert np.linalg.det(Rcw) - 1 < 1e-6, np.linalg.det(Rcw)

# f1 = (lower_bound + dz + np.random.uniform(0, 1e-5, 1)) * \
#     np.linalg.inv(R1) @ (projected1_xy)
# f0 = lower_bound * np.linalg.inv(R0) @ projected0_xy

# point_vels = (f1 - f0) / dt
# print(np.mean(point_vels, axis=1), gt[:3].T, np.std(point_vels, axis=1))

# # # plot a histogram of the point velocity distribution
# plt.hist(point_vels[0, :], bins=100, alpha=0.5)
# plt.hist(point_vels[1, :], bins=100, alpha=0.4)
# plt.hist(point_vels[2, :], bins=100, alpha=0.3)
# plt.show()


# # %%

# distances = np.linalg.norm(
#     point_vels - np.array([[vel, vel, vel]]).reshape(3, 1), axis=0, ord=2)
# # point_vels[:, np.argmin(distances)]
# good = np.where(distances < .5)
# # plt.scatter(projected1[0, good], projected1[1, good], c='r')
# # plt.scatter(projected1[0, :], projected1[1, :], c='g', alpha=1e-2)
# # plt.xlim(0, u0 * 2)
# # plt.ylim(0, v0 * 2)

# print(point_vels[:, good].reshape(3, -1).mean(axis=1))

# good_vel = point_vels[:, good].reshape(3, -1)
# print(np.linalg.norm(np.std(good_vel, axis=1), ord=1), good[0].shape)


# for i in range(int(1e5)):
#     random_good = np.random.choice(np.arange(0, N), 5)
#     good = (random_good,)
#     good_vel = point_vels[:, good].reshape(3, -1)
#     std = np.std(good_vel, axis=1)
#     if np.linalg.norm(std, ord=1) < .8:
#         print("found", std)
#         print(point_vels[:, good].reshape(3, -1).mean(axis=1))
#         break

# # # # plot a histogram of the point velocity distribution
# # plt.hist(good_vel[0, :], bins=100, alpha=0.5)
# # plt.hist(good_vel[1, :], bins=100, alpha=0.4)
# # plt.hist(good_vel[2, :], bins=100, alpha=0.3)
# # # plt.show()


# # %%

# w = np.max(projected0[0, :]) - np.min(projected0[0, :])
# h = np.max(projected0[1, :]) - np.min(projected0[1, :])
# slope = h/w


# DDD = 480 - (np.max(projected0[1, :]) - np.min(projected0[1, :]))


# diagonal0 = np.where(
#     np.abs(slope * projected0[0, :] - projected0[1, :] - DDD) < 1)

# projected1_xy = projected1_xy[:, diagonal0[0]]
# projected1 = projected1[:, diagonal0[0]]
# projected0_xy = projected0_xy[:, diagonal0[0]]
# projected0 = projected0[:, diagonal0[0]]

# plt.scatter(projected0[0, :], projected0[1, :], c='r')
# plt.xlim(0, u0 * 2)
# plt.ylim(0, v0 * 2)

# # ang_comp = skew_symmetric_matrix(ANGVEL)*dt + np.eye(3)
# Rcw = np.linalg.inv(T.R)
# assert np.linalg.det(Rcw) - 1 < 1e-6, np.linalg.det(Rcw)

# f1 = Rcw @ projected1_xy * (lower_bound + dz + np.random.uniform(0, 1e-5, 1))
# f0 = projected0_xy * (lower_bound)

# point_vels = (f1 - f0) / dt
# print(np.mean(point_vels, axis=1))

# # %%


# BEV


# %%

# %%
im0 = cv2.imread('1697640443.6579545.jpg')
im1 = cv2.imread('1697640443.6913054.jpg')

# define K
K = np.array([
    [285.0, 0.0, 320.0,],
    [0.0, 285.0, 240.0,],
    [0.0, 0.0, 1.0]
])
# define H
H = 10
# assume a an R matrix, just 45 degrees downward
R = np.array([
    [1.0, 0.0, 0.0,],
    [0.0, 0.70711, -0.70711,],
    [0.0, 0.70711, 0.70711,]
])

im_d = np.zeros(im0.shape[:2])
Kinv = la.inv(K)
u, v = np.meshgrid(range(0, im0.shape[1]), range(0, im0.shape[0]))
u = u.reshape(-1)
v = v.reshape(-1)
ones = np.ones_like(u)
Puv_hom = np.stack((u, v, ones), axis=-1)
Pc = Kinv @ Puv_hom.T
ls = R @ (Pc / la.norm(Pc, axis=0))
d = H / (np.array([[0, 0, -1]]) @ ls)
Pt = ls * d
distance = la.norm(Pt, axis=0)
distance = Pt[2, :]
im_d[v, u] = distance
# visualize image with color gradient as distance
plt.imshow(im_d, cmap='jet')
plt.colorbar()
plt.show()

# %%

every_nth = Pt[:, ::200]
# plot in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(every_nth[0, :], every_nth[1, :], every_nth[2, :])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# take a rectangle from these points
# and then create a perspective projection for birds eye view

py_max = Pt[1, :].max()
py_min = Pt[1, :].min()
px_max = Pt[0, :].max()
px_min = Pt[0, :].min()

# corners of rectangle
c0 = np.array([px_min, py_max, -H])
c1 = np.array([px_max, py_max, -H])
c2 = np.array([px_min, py_min, -H])
c3 = np.array([px_max, py_min, -H])
cs = []
for c in [c0, c1, c2, c3]:
    cp = K @ la.inv(R) @ c
    cp /= cp[2]
    cs.append(cp)

width = 640
height = 480
src_points = np.float32([[x, y] for x, y, _ in cs])
dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
M = cv2.getPerspectiveTransform(src_points, dst_points)
tim0 = cv2.warpPerspective(im0, M, (width, height))
plt.imshow(tim0)
plt.show()
plt.figure()
plt.imshow(im0)
plt.show()

tim1 = cv2.warpPerspective(im1, M, (width, height))
plt.figure()
plt.imshow(tim1)
plt.show()

# %%

# compute DIS flow on tim0 and tim1
tim0_gray = cv2.cvtColor(tim0, cv2.COLOR_BGR2GRAY)
tim1_gray = cv2.cvtColor(tim1, cv2.COLOR_BGR2GRAY)
disflow = cv2.DISOpticalFlow_create(1)

flow = disflow.calc(tim0_gray, tim1_gray, None)

hsv = np.zeros_like(im0)
hsv[..., 1] = 255
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# average of flow
flow_avg = np.mean(flow, axis=(0, 1))
plt.imshow(rgb)
plt.quiver(320, 240, flow_avg[0], flow_avg[1], color='r')
plt.show()

# %%

NTH = 16
plt.imshow(np.zeros_like(im0))
plt.quiver(np.arange(0, 640, NTH), np.arange(0, 480, NTH),
           flow[::NTH, ::NTH, 0], flow[::NTH, ::NTH, 1], color='r')
plt.show()

# %%

x0 = np.array([240, 320, 1], dtype=np.float32)
x1 = np.copy(x0)
x1[0] += flow[240, 320, 0]
x1[1] += flow[240, 320, 1]
p0 = (la.inv(M) @ x0) / (la.inv(M) @ x0)[-1]
p1 = (la.inv(M) @ x1) / (la.inv(M) @ x1)[-1]
p0, p1


def to3D(Puv_hom):
    Pc = Kinv @ Puv_hom.T
    ls = R @ (Pc / la.norm(Pc, axis=0))
    d = H / (np.array([[0, 0, -1]]) @ ls)
    Pt = ls * d
    return Pt


to3D(p1) - to3D(p0)
