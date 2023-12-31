# %%
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import re
from scipy import optimize
import transforms3d as tf


def e2h(e):
    if e.ndim == 1:
        return np.append(e, 1)

    return np.vstack((e, np.ones(e.shape[1])))


def get3D(pts, K, Rot, height):
    Kinv = la.inv(K)
    Pc = Kinv @ pts.T
    ls = Rot @ (Pc / la.norm(Pc, axis=0))
    d = height / (np.array([[0, 0, -1]]) @ ls)
    Pt = ls * d  # world points
    return Pt


def Lx(p_xy, Zs):
    if not isinstance(Zs, type(np.array)):
        Zs = np.ones_like(p_xy[:, 0]) * Zs

    Lx = np.zeros((p_xy.shape[0] * 2, 6))

    for i in range(p_xy.shape[0]):
        x = p_xy[i, 0]
        y = p_xy[i, 1]
        Z = Zs[i]

        Lx[2*i:2*i+2, :] = np.array([
            [-1/Z,  0,     x/Z, x * y,      -(1 + x**2), y,],
            [0,   -1/Z,   y/Z, (1 + y**2), -x*y,       -x]])

    return Lx


def Lp(p_uv, Zs, K):
    assert p_uv.shape[1] == 3
    # assert p_uv.shape[0] > 3

    if not isinstance(Zs, type(np.array)):
        Zs = np.ones_like(p_uv[:, 0]) * Zs

    Lx = np.zeros((p_uv.shape[0] * 2, 6))
    Kinv = la.inv(K)
    for i in range(p_uv.shape[0]):
        xy = Kinv @ p_uv[i, :]
        assert xy.shape == (3,)
        assert xy[2] == 1
        x = xy[0]
        y = xy[1]
        Z = Zs[i]

        Lx[2*i:2*i+2, :] = K[:2, :2] @ np.array([
            [-1/Z,  0,     x/Z, x * y,      -(1 + x**2), y,],
            [0,   -1/Z,   y/Z, (1 + y**2), -x*y,       -x]])

    return Lx


def h2e(h):
    if h.ndim == 1:
        return h[:-1]/h[-1]

    return h[:-1, :]/h[-1, :]


data = pd.read_csv('pjdata.csv')

# %%

TIMEPOINT = 412

velx = data["/fmu/out/vehicle_gps_position/vel_e_m_s"].dropna().to_numpy()
vely = data["/fmu/out/vehicle_gps_position/vel_n_m_s"].dropna().to_numpy()
velz = data["/fmu/out/vehicle_gps_position/vel_d_m_s"].dropna().to_numpy()
time = data["/fmu/out/vehicle_gps_position/timestamp"].dropna() / 1e6

# velx = data["/fmu/out/vehicle_odometry/velocity.0"].dropna().to_numpy()
# vely = data["/fmu/out/vehicle_odometry/velocity.1"].dropna().to_numpy()
# velz = data["/fmu/out/vehicle_odometry/velocity.2"].dropna().to_numpy()
# time = data["/fmu/out/vehicle_odometry/timestamp"].dropna() / 1e6


def getGT(timestamp):
    # find the time closest to TIMEPOINT seconds
    gt_vel = np.array([
        velx[np.argmin(np.abs(time - timestamp))],
        vely[np.argmin(np.abs(time - timestamp))],
        velz[np.argmin(np.abs(time - timestamp))],
    ])
    return gt_vel


plt.plot(time, velx, label="velx")
plt.plot(time, vely, label="vely")
plt.plot(time, velz, label="velz")
plt.legend()
plt.show()


def parse_meta(datafile):
    # Define patterns for single and multi-line values
    single_value_pattern = r"(\w+):\s*([-+]?\d*\.\d+|\d+)(?=\s*\w+:|$)"
    multi_value_pattern = r"(\w+):\s*([\d\s\.\-+eE]+)(?=\s*\w+:|$)"

    # Initialize dictionaries
    single_values_dict = {}
    multi_values_dict = {}

    # Find all single and multi-line values
    for name, value in re.findall(single_value_pattern, datafile):
        single_values_dict[name] = float(value)

    for name, value in re.findall(multi_value_pattern, datafile):
        multi_values_dict[name] = [float(val) for val in value.split()]

    # Combine the dictionaries
    parsed_data = {**single_values_dict, **multi_values_dict}

    # Convert values to numpy arrays and reshape if necessary
    for key, value in parsed_data.items():
        value_array = np.array(value)
        if value_array.size == 3:
            parsed_data[key] = value_array.reshape(-1, 1)
        elif value_array.size == 9:
            parsed_data[key] = value_array.reshape(3, 3)
        else:
            parsed_data[key] = value_array

    # Calculate 'dt'
    if "time" in parsed_data and "prev_time" in parsed_data:
        parsed_data["dt"] = parsed_data["time"] - parsed_data["prev_time"]

    return parsed_data


u, v = np.meshgrid(range(0, 640), range(0, 480))
u = u.reshape(-1)
v = v.reshape(-1)
ones = np.ones_like(u)
Puv_hom = np.stack((u, v, ones), axis=-1)


def getPt(K, Rot, height):
    global Puv_hom
    Kinv = la.inv(K)
    Pc = Kinv @ Puv_hom.T
    ls = Rot @ (Pc / la.norm(Pc, axis=0))
    d = height / (np.array([[0, 0, -1]]) @ ls)
    Pt = ls * d  # world points
    return Pt


def bev(image, Rot, height, jet=False, warp=False):
    Kinv = la.inv(K)

    Puv_hom = np.stack((u, v, ones), axis=-1)
    Pc = Kinv @ Puv_hom.T
    ls = Rot @ (Pc / la.norm(Pc, axis=0))
    d = height / (np.array([[0, 0, -1]]) @ ls)
    Pt = ls * d  # world points

    if jet:
        distance = la.norm(Pt, axis=0)
        distance = Pt[0, :]
        im_d = np.zeros(image.shape[:2])
        im_d[v, u] = distance
        # visualize image with color gradient as distance
        plt.imshow(im_d, cmap='jet')
        plt.colorbar()
        plt.show()

    py_max = Pt[1, :].max()
    py_min = Pt[1, :].min()
    px_max = Pt[0, :].max()
    px_min = Pt[0, :].min()
    # corners of rectangle
    c0 = np.array([px_min, py_max, -H])
    c1 = np.array([px_max, py_max, -H])
    c2 = np.array([px_min, py_min, -H])
    c3 = np.array([px_max, py_min, -H])

    # def is_rect(diagonal0, diagonal1):
    #     return abs(diagonal0 - diagonal1) < 1e-2
    # print("Is rectangle? ",
    #       is_rect(la.norm(c0 - c3), la.norm(c1 - c2)))

    cs = []
    for c in [c1, c3, c0, c2]:
        cp = K @ la.inv(Rot) @ c
        cp /= cp[2]
        cs.append(cp)

    width = 640
    height = 480
    src_points = np.float32([[x, y] for x, y, _ in cs])
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    tim0 = cv2.warpPerspective(im0, M, (width, height))
    # crop zeros
    tim0 = tim0[~np.all(tim0 == 0, axis=1)]
    tim0 = tim0[:, ~np.all(tim0 == 0, axis=0)]
    if warp:
        plt.imshow(tim0, cmap='gray')
        plt.show()
        plt.figure()
        plt.imshow(im0, cmap='gray')
        plt.show()
    return tim0, Pt, M


saved = os.listdir('/tmp/')
saved.sort()

# %%


def extract_z_rot(R):
    z = np.array([R[0, 2], R[1, 2], R[2, 2]])
    z /= la.norm(z)
    theta = np.arctan2(z[1], z[0])
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]])
    return Rz


def get_timestamp(f):
    parts = f.split("_")
    try:
        ts = float(parts[0])
    except ValueError:
        return None
    return ts


disflow = cv2.DISOpticalFlow_create(2)
T0 = 440
meas_vel = []
meas_time = []
ts_spot = None
for i in range(len(saved)):
    try:
        sample = saved[i]
    except IndexError:
        continue
    tmp = get_timestamp(sample)
    if tmp is None:
        continue
    if ts_spot is None:
        ts_spot = tmp
    if tmp == ts_spot:
        continue
    ts_spot = tmp

    if ts_spot < T0:
        continue

    # read these in at ts_spot
    im0 = cv2.imread(f'/tmp/{ts_spot:.6f}_frame0.png', cv2.IMREAD_GRAYSCALE)
    im1 = cv2.imread(f'/tmp/{ts_spot:.6f}_frame1.png', cv2.IMREAD_GRAYSCALE)
    flowdatafile = open(f'/tmp/{ts_spot:.6f}_flowinfo.txt')
    flowdata = flowdatafile.read()

    parsed_data = parse_meta(flowdata)
    if not isinstance(parsed_data, dict):
        continue
    K = parsed_data["K"]
    omega_cam = parsed_data["omega"]
    omega_drone = parsed_data["drone_omega"]
    H = float(parsed_data["height"][0])
    pH = float(parsed_data["prev_height"][0])
    R = parsed_data["cam_R_enu"]
    prevR = parsed_data["prev_R"]
    dt = parsed_data["dt"]

    bev0, _, H0 = bev(im0, prevR, pH, warp=False)
    bev1, _, H1 = bev(im1, R, H, warp=False)
    flow = disflow.calc(bev0, bev1, None)  # (480, 640, 2)
    flow = flow.reshape(-1, 2)
    NTH = 7
    pixels = np.stack((u, v, ones), axis=-1)
    filter_zeros = np.where(bev0[pixels[:, 1], pixels[:, 0]] > 0)
    pixels = pixels[filter_zeros][::NTH**2, :]
    flow = flow[filter_zeros][::NTH**2, :]

    Kinv = la.inv(K)
    Jac = Lp(pixels, H, K)
    Jac = Jac[:, [1, 2, 5]]
    velocity = la.pinv(Jac) @ (flow.reshape(-1, 1) / dt)
    # Rz = extract_z_rot(R)
    # velocity[:3] = la.inv(Rz) @ velocity[:3]
    meas_vel.append([velocity[0][0], velocity[1][0],
                     (la.norm(omega_drone) + la.norm(omega_cam))])

    # flow_avg = np.mean(flow, axis=0)
    # pp = np.array([320, 240, 1], dtype=np.float32)
    # pp_org = h2e(la.inv(H0) @ pp)
    # pp_flow = np.copy(pp)
    # pp_flow[:2] += flow_avg
    # pp_flow_org = h2e(la.inv(H1) @ pp_flow)
    # dp = get3D(e2h(pp_org), K, prevR, pH) - get3D(e2h(pp_flow_org), K, R, H)
    # velocity = dp / dt
    # meas_vel.append([velocity[0], velocity[1],
    #                  (la.norm(omega_drone) + la.norm(omega_cam))])

    meas_time.append(ts_spot)

# %%

meas_vel_np = np.array(meas_vel)
# meas_vel_np[:, 0] = pd.Series(meas_vel_np[:, 0]).rolling(30).mean()
# meas_vel_np[:, 1] = pd.Series(meas_vel_np[:, 1]).rolling(30).mean()
plt.figure()
plt.xlim(T0, 442.1)
plt.ylim(-5, 5)
plt.plot(meas_time, meas_vel_np[:, 0], label="meas velx")
plt.plot(meas_time, meas_vel_np[:, 1], label="meas vely")
plt.plot(time, velx, label="gt velx")
plt.plot(time, vely, label="gt vely")
plt.legend()

plt.figure()
plt.xlim(T0, 442)
plt.ylim(0, 2)
plt.plot(time, np.sqrt(velx**2 + vely**2), label="NORM GT")
plt.plot(meas_time, np.sqrt(meas_vel_np[:, 0]**2 + meas_vel_np[:, 1]**2),
         label="NORM MEAS")
plt.plot(meas_time, meas_vel_np[:, 2], label="omega")
plt.legend()
plt.show()

# %%


def objective(x, hom, rt):
    # H = K [r1​,r2​,t] solve for K
    cam = x.reshape(3, 3)
    return la.norm(hom - cam @ rt)


def decomposeKFromH(homography, cam_mat):
    # H = K [r1​,r2​,t] solve for K
    num, Rs, Ts, _ = cv2.decomposeHomographyMat(homography, cam_mat)
    for i in range(num):
        rt = np.hstack((Rs[i][:, :2], Ts[i]))
        res = optimize.minimize(objective, cam_mat.reshape(-1),
                                args=(homography, rt),
                                method='Nelder-Mead')
        print(res.x.reshape(3, 3) / res.x[8],
              res.fun, res.success)

        return res.x.reshape(3, 3) / res.x[8]


Kcom = decomposeKFromH(H1, K)

h2e(la.inv(Kcom) @ pp)

# %%


def get_timestamp(f):
    parts = f.split("_")
    try:
        ts = float(parts[0])
    except ValueError:
        return None
    return ts


T0 = 440
ts_spot = None
for i in range(10000):
    try:
        sample = saved[i]
    except IndexError:
        break
    ts = get_timestamp(sample)
    if ts is None:
        continue
    if ts - T0 < (30/1000):
        ts_spot = ts

# read these in at ts_spot
im0 = cv2.imread(f'/tmp/{ts_spot:.6f}_frame0.png', cv2.IMREAD_GRAYSCALE)
im1 = cv2.imread(f'/tmp/{ts_spot:.6f}_frame1.png', cv2.IMREAD_GRAYSCALE)
flow = open(f'/tmp/{ts_spot:.6f}_flowinfo.txt')
flowdata = flow.read()

parsed_data = parse_meta(flowdata)
K = parsed_data["K"]
omega_cam = parsed_data["omega"]
omega_drone = parsed_data["drone_omega"]
print(f"omegas {omega_cam.T} {omega_drone.T}")
H = float(parsed_data["height"][0])
pH = float(parsed_data["prev_height"][0])
R = parsed_data["cam_R_enu"]
prevR = parsed_data["prev_R"]
dt = parsed_data["dt"]

bev0, Pt0, H0 = bev(im0, prevR, pH, warp=False)
bev1, Pt1, H1 = bev(im1, R, pH, warp=False)


def modcont(image):
    blurred_img = cv2.GaussianBlur(image, (21, 21), 0)
    mask = np.zeros(image.shape, np.uint8)
    thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255), 5)
    output = np.where(mask == np.array([255]), blurred_img, image)
    return output


bev0 = modcont(bev0)
bev1 = modcont(bev1)

disflow = cv2.DISOpticalFlow_create(2)
flow = disflow.calc(bev0, bev1, None)  # (480, 640, 2)

NTH = 15
# pixels = np.stack((u, v, ones), axis=-1)
# flow = flow.reshape(-1, 2)
pixels = np.stack((u, v, ones), axis=-1)
flow = flow.reshape(-1, 2)
filter_zeros = np.where(bev0[pixels[:, 1], pixels[:, 0]] > 0)
pixels = pixels[filter_zeros][::NTH**2, :]
flow = flow[filter_zeros][::NTH**2, :]

# if True:
plt.scatter(pixels[:, 0], pixels[:, 1], s=1)
# invert y axis
plt.gca().invert_yaxis()
plt.figure()
plt.imshow(bev0)
plt.quiver(pixels[:, 0], pixels[:, 1],
           flow[:, 1]*10, flow[:, 0]*10, color='r',
           angles='xy', scale_units='xy', scale=1)
flow_avg = np.mean(flow, axis=0)
plt.quiver(320, 240, flow_avg[1]*100, flow_avg[0]*100, color='b',
           angles='xy', scale_units='xy', scale=1)
plt.show()

Kinv = la.inv(K)
Jac = Lp(pixels, H, K)
velw = la.pinv(Jac) @ (flow.reshape(-1, 1) / dt)
velw, getGT(ts_spot), ts_spot

# %%

pp = np.array([320, 240, 1], dtype=np.float32)
pp_org = h2e(la.inv(H0) @ pp)

pp_flow = np.copy(pp)
pp_flow[:2] += flow_avg
pp_flow_org = h2e(la.inv(H1) @ pp_flow)
dp = get3D(e2h(pp_org), K, prevR, pH) - get3D(e2h(pp_flow_org), K, R, H)

v = dp / dt
v, getGT(ts_spot)

# np.set_printoptions(precision=3, suppress=True)
# back = la.inv(K @ H0) @ pixels.T
# back /= back[2, :]

# # plt.scatter(back[0, :], back[1, :], s=1)
# # plt.gca().invert_yaxis()

# H0 @ pixels.T, la.inv(K @ H0)

# Kb = H0 @ la.inv(K)
# pixels_b = Kb @ pixels.T
# pixels_b /= pixels_b[2, :]
# plt.scatter(pixels_b[0, :], pixels_b[1, :], s=1)
# plt.gca().invert_yaxis()

# %%


pixels_in_org = h2e(la.inv(H0) @ pixels.T)
pixels_in_org = e2h(pixels_in_org)
pts0 = get3D(pixels_in_org.T, K, prevR, pH)

pixels_flow = np.copy(pixels).astype(np.float32)
pixels_flow[:, :2] += flow
pixels_in_org1 = h2e(la.inv(H1) @ pixels_flow.T)
pixels_in_org1 = e2h(pixels_in_org1)
pts1 = get3D(pixels_in_org1.T, K, R, H)

vels = (pts0 - pts1) / dt

plt.figure()
plt.hist(vels[1, :], bins=100, color='r', alpha=0.2)
plt.hist(vels[0, :], bins=100, color='g', alpha=0.5)
plt.show()

vels = np.round(vels, 1)
vels_unique_x = np.unique(vels[0, :], return_counts=True)
vel_x = vels_unique_x[0][vels_unique_x[1].argmax()]
vels_unique_y = np.unique(vels[1, :], return_counts=True)
vel_y = vels_unique_y[0][vels_unique_y[1].argmax()]
vel_x, vel_y, gt_vel, vels.shape

# %%

# flow is in bev
# pixels are in original image
idxs = np.random.randint(0, pixels_flow.shape[0], 10)
# pixels_flow_sel = pixels_flow[idxs, :]
# flow_sel = flow[idxs, :]
pixels_sel = pixels
flow_sel = flow
Kinv = la.inv(K)
bev_pixels = h2e(H1 @ Kinv @ pixels_sel.T)
Jac = Lx((bev_pixels).T, H)
Jac = Jac[:, [0, 1, 5]]


def cost(K):
    K = K.reshape(3, 3)
    Kinv = la.inv(K)
    bev_pixels = h2e(H1 @ Kinv @ pixels_sel.T)
    Jac = Lx((bev_pixels).T, H)
    Jac = Jac[:, [0, 1, 5]]
    velw = la.pinv(Jac) @ (flow_sel.reshape(-1, 1) / dt)
    velw = velw.reshape(1, 3)
    gt = gt_vel[:2]
    gt = np.append(gt, 0)
    return la.norm(velw - gt, ord=1)


res = optimize.minimize(cost, K.reshape(-1))

# %%

Ksolved = np.array(res.x).reshape(3, 3)
Kinv = la.inv(Ksolved)
bev_pixels = h2e(H1 @ Kinv @ pixels_sel.T)
Jac = Lx((bev_pixels).T, H)
Jac = Jac[:, [0, 1, 5]]
velw = la.pinv(Jac) @ (flow_sel.reshape(-1, 1) / dt)
velw = velw.reshape(1, 3)
gt = gt_vel[:2]
gt = np.append(gt, 0)
velw, res

# %%


# Load the grayscale image
image = bev0

# Define a 3x3 kernel with all ones
kernel = np.ones((3, 3), np.uint8)

# Erode the image: expanding the black areas
eroded_image = cv2.erode(image, kernel, iterations=1)

plt.imshow(eroded_image, cmap='gray')
plt.show()
plt.imshow(bev0, cmap='gray')

# pixels_flow.shape
# flow.max()

# print(Pt1.shape, bev1.shape)

# points0_3d = Pt0[:, filter_zeros][:, :, ::NTH**2].squeeze()
# points1_3d = Pt1[:, filter_zeros][:, :, ::NTH**2].squeeze()

# # plot in 3d
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(points0_3d[0, :], points0_3d[1, :], points0_3d[2, :])
# ax.scatter(points1_3d[0, :], points1_3d[1, :], points1_3d[2, :], color='r')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

# vels = (points1_3d - points0_3d) / dt

# # plt.figure()
# # # do histogram of velocities
# # # plt.hist(vels[0, :], bins=1000, color='b', alpha=0.5)
# # plt.hist(vels[1, :], bins=40, color='r', alpha=0.5)
# # plt.show()

# # print most common velocity
# vels_x = np.round(vels, 2)
# vels_unique_x = np.unique(vels_x[0, :], return_counts=True)
# print(vels_unique_x[0][vels_unique_x[1].argmax()], gt_vel[0])
# vels_y = np.round(vels, 2)
# vels_unique_y = np.unique(vels_y[1, :], return_counts=True)
# print(vels_unique_y[0][vels_unique_y[1].argmax()], gt_vel[1])

# %%
# def costK(x, pts, K, H0):
#     Kinv = la.inv(K)
#     Puv_hom = np.stack((u, v, ones), axis=-1)
#     Pc = H0 @ Kinv @ Puv_hom.T
#     Pc = Pc[:, filter_zeros].squeeze()[:, ::NTH**2]
#     Kb = x.reshape(3, 3)
#     xb = la.inv(Kb) @ pts.T
#     return la.norm(Pc - xb, ord=1) + la.norm(Kb[2, 2] - 1)

#     return la.norm(xb - Pc)

#     Kinv = la.inv(K)
#     Pc = Kinv @ pts.T
#     Pt = Pc * Z

#     back = K @ Pt
#     back /= back[2, :]
#     return la.norm(pts - back.T)


# costK(np.eye(3), pixels, K, H0)
# res = optimize.minimize(costK, np.eye(3).reshape(-1), args=(pixels, K, H0))
# res

# !!!FROM LOOP!!!
# NTH = 13
# Pt0 = getPt(K, prevR, pH)[:, ::NTH**2]
# Pt1 = getPt(K, R, H)[:, ::NTH**2]

# vels = (Pt0 - Pt1) / dt
# vels = np.round(vels, 2)

# # plt.figure()
# # plt.hist(vels[0, :], bins=100, color='b', alpha=0.5)
# # plt.hist(vels[1, :], bins=100, color='r', alpha=0.5)
# # plt.show()

# # # print most common velocity
# vels_unique_x = np.unique(vels[0, :], return_counts=True)
# vel_x = vels_unique_x[0][vels_unique_x[1].argmax()]
# vels_unique_y = np.unique(vels[1, :], return_counts=True)
# vel_y = vels_unique_y[0][vels_unique_y[1].argmax()]
