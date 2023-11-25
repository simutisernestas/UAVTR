# %%
import transforms3d as tf
from scipy.optimize import minimize, least_squares
from sklearn.linear_model import LinearRegression
import spatialmath
import machinevisiontoolbox as mv
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
DEG2RAD = np.pi/180
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
RAD2DEG = 180/np.pi


def getcube(n):
    d_x = d_y = d_z = 1 / n
    x0 = y0 = z0 = -0.5
    x = np.arange(x0, -x0, d_x, dtype=float)
    y = np.arange(y0, -y0, d_y, dtype=float)
    z = np.arange(z0, -z0, d_z, dtype=float)
    x = np.append(x, 0.5)
    y = np.append(y, 0.5)
    z = np.append(z, 0.5)
    cube = np.stack(np.meshgrid(x, y, z))
    Q = cube.reshape(3, -1)  # cube
    Q = np.swapaxes(Q, 0, 1)
    Q = Q[(abs(Q) == 0.5).sum(axis=1) >= 2]
    ones = np.ones(Q.shape[0]).reshape(Q.shape[0], 1)
    Q = np.concatenate((Q, ones), axis=1)
    Q = np.append(np.vstack(
        (x, np.zeros_like(x), np.zeros_like(x), np.ones_like(x))).T, Q, axis=0)
    Q = np.append(np.vstack(
        (np.zeros_like(x), y, np.zeros_like(x), np.ones_like(x))).T, Q, axis=0)
    Q = np.append(np.vstack(
        (np.zeros_like(x), np.zeros_like(x), z, np.ones_like(x))).T, Q, axis=0)
    return Q


def make_image_from_3D_points(camera, points, flip=True):
    projected, _ = camera.project_point(points, visibility=True)
    img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    WHICH = 1
    min_height = np.min(points[WHICH, :])
    max_height = np.max(points[WHICH, :])
    for i in range(projected.shape[1]):
        p = projected[:, i]
        if np.isnan(p[0]) or np.isnan(p[1]):
            continue
        u, v = int(p[1]), int(p[0])
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if u+j < 0 or u+j >= IMAGE_HEIGHT or v+k < 0 or v+k >= IMAGE_WIDTH:
                    continue
                # make points higher is space blue
                height = points[WHICH, i].copy()
                color = (height - min_height) / (max_height - min_height)
                # map color from height to 0..255
                img[u+j, v+k] = (color, 0, 255)
    if flip:
        img = np.flip(img, axis=0)
        # img = np.flip(img, axis=1)
    plt.figure(dpi=300)
    plt.imshow(img)
    plt.xlabel('u (px)')
    plt.ylabel('v (px)')
    return img


Q = getcube(16)
pose = spatialmath.SE3.Tz(-1.5) * spatialmath.SE3.Ty(1.5) * \
    spatialmath.SE3.Rx(45*DEG2RAD)
cam = mv.CentralCamera(f=385,
                       pp=(320, 240),
                       imagesize=(IMAGE_WIDTH, IMAGE_HEIGHT),
                       pose=pose,
                       name='my camera')

make_image_from_3D_points(cam, Q.T[:3, :])
del cam

# %%


def make_terrain(bound=20, max_height=3, spacing=2.0):
    # make terain of points laying on the ground plane with random height
    Q_terrain = np.zeros((3, 0))
    mesh_grid_xy = np.arange(-bound, bound+spacing, spacing)
    x, y = np.meshgrid(mesh_grid_xy, mesh_grid_xy)
    z = np.random.uniform(0, max_height, size=x.shape)
    Q_terrain = np.append(Q_terrain, np.array(
        [y.flatten(), z.flatten(), x.flatten()]), axis=1)
    Q_terrain = Q_terrain.T
    return Q_terrain


def make_cameras_with_motion(int_motion, h=40, noise=1e-1):  # 30Hz
    pose = spatialmath.SE3.RPY(45*DEG2RAD, 0*DEG2RAD, 0*DEG2RAD)
    pose.t = np.array([0, h, -h])

    f = 8*1e-3     # focal length in metres
    rho = 10*1e-6  # pixel side length in metres
    u0 = 320       # principal point, horizontal coordinate
    v0 = 240       # principal point, vertical coordinate
    camera1 = mv.CentralCamera(f=f, rho=rho, pp=(u0, v0),
                               imagesize=(IMAGE_WIDTH, IMAGE_HEIGHT),
                               pose=pose, noise=noise)

    # switch int_motion 3 with 5
    int_motion[3], int_motion[5] = int_motion[5], int_motion[3]

    move = spatialmath.SE3.Tz(int_motion[0]) * \
        spatialmath.SE3.Ty(int_motion[1]) * \
        spatialmath.SE3.Tz(int_motion[2]) * \
        spatialmath.SE3.RPY(int_motion[3:], order='xyz')
    camera2 = camera1.move(move, relative=True)

    return camera1, camera2


dt = 1/33
velocity = np.random.uniform(-1, 1, size=3)
angular = np.random.uniform(-np.pi/2, np.pi/2, size=3)
MOTION = np.concatenate([velocity, angular]) * dt
camera1, camera2 = make_cameras_with_motion(MOTION, h=40, noise=1e-6)
Q_terrain = make_terrain(bound=13, max_height=10, spacing=1.0)
img = make_image_from_3D_points(camera1, Q_terrain.T, flip=True)
make_image_from_3D_points(camera2, Q_terrain.T, flip=True)
# switch int_motion 3 with 5
MOTION[3], MOTION[5] = MOTION[5], MOTION[3]

# %%


def solve_for_vel(Qs, camera1, camera2):
    # solve for velocity using 3D points
    features = np.zeros((2, Qs.shape[0], 2))
    feature_depth = np.zeros((2, Qs.shape[0]))
    for i, it_camera in enumerate([camera1, camera2]):
        projected, vis = it_camera.project_point(Qs.T, visibility=True)
        assert np.all(vis)
        features[i] = projected.T
        depths = np.linalg.norm(Qs - it_camera.pose.t, axis=1)
        feature_depth[i, :] = depths  # depths  # / depths.max()

    Jacobian = camera2.visjac_p(features[1].T, feature_depth[1])
    flow = (features[1] - features[0])

    x = np.linalg.pinv(Jacobian) @ flow.reshape((-1, 1))
    # x, _, _, _ = np.linalg.lstsq(
    #     Jacobian, flow.reshape((-1, 1)), rcond=None)
    return x, (Jacobian, flow, features, feature_depth)


x, _ = solve_for_vel(Q_terrain, camera1, camera2)
print(np.linalg.norm(x - MOTION.reshape((-1, 1))))

# %%

ZERO_OUT = False
N_SIMS = 1000
errors = np.zeros((N_SIMS, 6))
for i in range(N_SIMS):
    Q_terrain = make_terrain(bound=10, max_height=3, spacing=2.5)
    # zero out Z
    if ZERO_OUT:
        Q_terrain[:, 2] = 0
    dt = 1/33
    velocity = np.random.uniform(-1, 1, size=3)
    angular = np.random.uniform(-np.pi/2, np.pi/2, size=3)
    MOTION = np.concatenate([velocity, angular]) * dt
    camera1, camera2 = make_cameras_with_motion(MOTION, h=40, noise=.1)
    x, _ = solve_for_vel(Q_terrain, camera1, camera2)
    MOTION[3], MOTION[5] = MOTION[5], MOTION[3]
    errors[i] = x.ravel() - MOTION

# make statistics of error along every dimension
plt.figure(dpi=300)
plt.boxplot(errors[:, :3])
plt.xticks(np.arange(1, 4), ['vx', 'vy', 'vz'])
plt.ylabel('error (m/s)')
None
plt.figure(dpi=300)
plt.boxplot(errors[:, 3:])
plt.xticks(np.arange(1, 4), ['wx', 'wy', 'wz'])
plt.ylabel('error (m/s)')

# %%

x, (Jacobian, flow, features, feature_depth) = solve_for_vel(
    Q_terrain, camera1, camera2)

# plot features
plt.figure(dpi=300)
plt.scatter(features[0, :, 0], features[0, :, 1], color='b', alpha=0.5)
plt.scatter(features[1, :, 0], features[1, :, 1], color='r', alpha=0.5)
plt.xlabel('u (px)')
plt.ylabel('v (px)')
plt.xlim(0, IMAGE_WIDTH)
plt.ylim(0, IMAGE_HEIGHT)
plt.quiver(features[0, :, 0], features[0, :, 1],
           flow[:, 0], flow[:, 1], color='b', alpha=0.5,
           angles='xy', scale_units='xy', scale=1)
# plt.gca().invert_yaxis()
None

camera1.flowfield(MOTION)

# %%

# computed from jacobian
px_flows = np.matmul(Jacobian, MOTION)
px_flows = px_flows.reshape((-1, 2))
# px_flows[:, 1] *= -1

plt.figure(dpi=300)
for i in range(0, Q_terrain.shape[0]):
    plt.quiver(features[0, i, 0], features[0, i, 1],
               flow[i, 0], flow[i, 1], color='b', alpha=0.5,
               angles='xy', scale_units='xy', scale=1)
    plt.quiver(features[0, i, 0], features[0, i, 1],
               px_flows[i, 0], px_flows[i, 1], color='r', alpha=0.5,
               angles='xy', scale_units='xy', scale=1)
# increase xy
plt.xlim(0, IMAGE_WIDTH)
plt.ylim(0, IMAGE_HEIGHT)
plt.xlabel('u (px)')
plt.ylabel('v (px)')
# None