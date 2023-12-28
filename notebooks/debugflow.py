# %%
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import re
from scipy import optimize


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


data = pd.read_csv('pjdata.csv')

# %%

TIMEPOINT = 414

velx = data["/fmu/out/vehicle_odometry/velocity.0"].dropna().to_numpy()
vely = data["/fmu/out/vehicle_odometry/velocity.1"].dropna().to_numpy()
velz = data["/fmu/out/vehicle_odometry/velocity.2"].dropna().to_numpy()
time = data["/fmu/out/vehicle_odometry/timestamp"].dropna() / 1e6

# find the time closest to TIMEPOINT seconds
gt_vel = np.array([
    velx[np.argmin(np.abs(time - TIMEPOINT))],
    vely[np.argmin(np.abs(time - TIMEPOINT))],
    velz[np.argmin(np.abs(time - TIMEPOINT))],
])
print(gt_vel)

plt.plot(time, velx, label="velx")
plt.plot(time, vely, label="vely")
plt.plot(time, velz, label="velz")
plt.legend()
plt.show()

saved = os.listdir('/tmp/')
saved.sort()

ts_spot = None
for f in saved:
    parts = f.split("_")
    try:
        ts = float(parts[0])
    except ValueError:
        continue
    if abs(ts - TIMEPOINT) < 2e-2:
        ts_spot = ts
assert ts_spot is not None

# read these in at ts_spot
im0 = cv2.imread(f'/tmp/{ts_spot:.6f}_frame0.png', cv2.IMREAD_GRAYSCALE)
im1 = cv2.imread(f'/tmp/{ts_spot:.6f}_frame1.png', cv2.IMREAD_GRAYSCALE)
# plt.imshow(im0, cmap='gray')
# plt.figure()
# plt.imshow(im1, cmap='gray')

flow = open(f'/tmp/{ts_spot:.6f}_flowinfo.txt')
flowdata = flow.read()

# Define the patterns for the different types of data
single_value_pattern = r"(\w+):\s*([-+]?\d*\.\d+|\d+)(?=\s*\w+:|$)"
multi_value_pattern = r"(\w+):\s*((?:[-+]?\d*\.\d+|\d+\s*)+)(?=\s*\w+:|$)"

# Find all the single values
single_values = re.findall(single_value_pattern, flowdata)

# Find all the multi values
multi_values = re.findall(multi_value_pattern, flowdata)

# Convert the single values to a dictionary
single_values_dict = {name: float(value) for name, value in single_values}

# Convert the multi values to a dictionary with lists of floats
multi_values_dict = {name: [float(val) for val in value.split()]
                     for name, value in multi_values}

# Combine the dictionaries
parsed_data = {**single_values_dict, **multi_values_dict}

# convert the values to numpy arrays
for key, value in parsed_data.items():
    parsed_data[key] = np.array(value)
    if parsed_data[key].shape[0] == 3:
        parsed_data[key] = parsed_data[key].reshape(-1, 1)
    if parsed_data[key].shape[0] == 9:
        parsed_data[key] = parsed_data[key].reshape(3, 3)
    print(key, parsed_data[key])
parsed_data["dt"] = parsed_data["time"] - parsed_data["prev_time"]

K = parsed_data["K"]
H = parsed_data["height"]
R = parsed_data["cam_R_enu"]
prevR = parsed_data["prev_R"]
dt = parsed_data["dt"]

im_d = np.zeros(im0.shape[:2])
Kinv = la.inv(K)
u, v = np.meshgrid(range(0, im0.shape[1]), range(0, im0.shape[0]))
u = u.reshape(-1)
v = v.reshape(-1)
ones = np.ones_like(u)
Puv_hom = np.stack((u, v, ones), axis=-1)
Pc = Kinv @ Puv_hom.T
ls = prevR @ (Pc / la.norm(Pc, axis=0))
d = H / (np.array([[0, 0, -1]]) @ ls)
Pt = ls * d
Pt = la.inv(prevR) @ Pt
distance = la.norm(Pt, axis=0)
distance = Pt[2, :]
im_d[v, u] = distance
plt.imshow(im_d, cmap='jet')
plt.colorbar()
plt.show()

disflow = cv2.DISOpticalFlow_create(2)
flow = disflow.calc(im0, im1, None)

NTH = 13
pixels = np.stack((u, v, ones), axis=-1)[::NTH**2, :]
flow = flow.reshape(-1, 2)[::NTH**2, :]
Z = Pt[2, :][::NTH**2]

if True:
    plt.scatter(pixels[:, 0], pixels[:, 1], s=1)
    plt.figure()
    plt.imshow(im1)
    plt.quiver(pixels[:, 0], pixels[:, 1],
               flow[:, 0]*10, flow[:, 1]*10, color='r',
               angles='xy', scale_units='xy', scale=1)
    flow_avg = np.median(flow, axis=0)
    plt.quiver(320, 240, flow_avg[0]*10, flow_avg[1]*10, color='g',
               angles='xy', scale_units='xy', scale=1)
    plt.show()

# filter = np.where(la.norm(flow, axis=1) > 6)[0]
# filter = np.where(Z < 30)
# Z = Z[filter]
# pixels = pixels[filter]
# flow = flow[filter]

print(f"Shape of pixels: {pixels.shape}")
print(f"Shape of flow: {flow.shape}")
print(f"Z shape: {Z.shape}")

Jac = Lp(pixels, Z, K)
print(f"Jac shape: {Jac.shape}")
dflow = flow.reshape(-1, 1) / dt
print(f"Shape of dflow: {dflow.shape}")

cam_omega = parsed_data["omega"]
drone_omega = parsed_data["drone_omega"]
arm = parsed_data["r"]
print(f"Arm {arm.T}")
print(f"Drone omega {drone_omega.T}")
print(f"Cam omega {cam_omega.T}")

dflow -= Jac[:, 3:] @ cam_omega

vel = np.linalg.pinv(Jac)[:3, :] @ dflow
print(f"Enu vel: {(R @ vel[:3]).T}")

gv = gt_vel.reshape(-1, 1)
gv_enu = np.array([[gv[1, 0], gv[0, 0], -gv[2, 0]]]).T
err = la.norm((R @ vel[:3] - gv_enu)[:2], ord=1)
print(f"GT: {gv_enu.T}")
print(f"Norm error: {err}")

# def cost(x):
#     arm_in = x[:3].reshape(3, 1)
#     Rvel = x[3:].reshape(3, 3)
#     velocity = vel[:3]
#     induced = np.cross(drone_omega.T, arm_in.T).T
#     loss = la.norm((R @ (velocity - induced) - gt_vel.reshape(-1, 1)))
#     # loss += la.norm(Rvel @ Rvel.T - np.eye(3) + 1e-6)
#     # ensure arm length is maximum 0.15 meters
#     loss += la.norm(arm_in) * 5e-2
#     return loss

# x = np.concatenate((arm.reshape(-1), np.eye(3).reshape(-1)))
# print(cost(x))
# res = optimize.minimize(cost, x)
# cost(res.x), res.x[:3], res.x[3:].reshape(
#     3, 3), la.det(res.x[3:].reshape(3, 3))

# NED - North East Down
# ENU - East North Up

# %%
