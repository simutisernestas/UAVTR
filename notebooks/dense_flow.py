# %%
import scipy
from scipy.optimize import minimize
import spatialmath
from scipy.spatial.transform import Rotation
import machinevisiontoolbox as mv
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
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
        img = np.flip(img, axis=1)
    plt.figure(dpi=200)
    plt.imshow(img)
    plt.xlabel('u (px)')
    plt.ylabel('v (px)')


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

# make terain of points laying on the ground plane with random height
Q_terrain = np.zeros((3, 0))

BOUND = 20
MAX_HEIGHT = 3
SPACING = 2.0
mesh_grid_xy = np.arange(-BOUND, BOUND+SPACING, SPACING)
x, y = np.meshgrid(mesh_grid_xy, mesh_grid_xy)
# np.random.uniform(-MAX_HEIGHT, MAX_HEIGHT, size=x.shape)
z = np.ones_like(x) * MAX_HEIGHT
Q_terrain = np.append(Q_terrain, np.array(
    [x.flatten(), z.flatten(), y.flatten()]), axis=1)
Q_terrain = Q_terrain.T

NOISE = 1e-1
pose = spatialmath.SE3.RPY(45*DEG2RAD, 0*DEG2RAD, 0*DEG2RAD)
pose.t = np.array([0, 30, -40])
camera1 = mv.CentralCamera(f=385,
                           pp=(320, 240),
                           imagesize=(IMAGE_WIDTH, IMAGE_HEIGHT),
                           pose=pose,
                           distortion=None,
                           name='cam1', noise=NOISE)
make_image_from_3D_points(camera1, Q_terrain.T, flip=True)

dt = 1/33
velocity = np.random.uniform(-3, 3, size=3)
angular_velocity = np.random.uniform(-np.pi/2, np.pi/2, size=3)
MOTION = np.concatenate([velocity, angular_velocity]) * dt
pose2 = pose * spatialmath.SE3.Tx(MOTION[0]) * \
    spatialmath.SE3.Ty(MOTION[1]) * \
    spatialmath.SE3.Tz(MOTION[2]) * \
    spatialmath.SE3.RPY(MOTION[3:])
camera2 = mv.CentralCamera(f=385,
                           pp=(320, 240),
                           imagesize=(IMAGE_WIDTH, IMAGE_HEIGHT),
                           pose=pose2,
                           distortion=None,
                           name='cam2', noise=NOISE)
make_image_from_3D_points(camera2, Q_terrain.T, flip=True)

# %%

N = 2
features = np.zeros((N, Q_terrain.shape[0], 2))
feature_depth = np.zeros((N, Q_terrain.shape[0]))

for i, it_camera in enumerate([camera1, camera2]):
    projected, vis = it_camera.project_point(Q_terrain.T, visibility=True)
    assert np.all(vis)
    features[i] = projected.T
    feature_depth[i, :] = np.linalg.norm(
        Q_terrain[:, :3] - it_camera.pose.t, axis=1)

Jacobian = camera2.visjac_p(features[1].T, feature_depth[1])
flow = features[1] - features[0]

# solve with pseudo inverse
x = np.linalg.pinv(Jacobian) @ flow.reshape((-1, 1))
print(np.linalg.norm(x - MOTION.reshape((-1, 1))))


def cost(x, Jac, Flow):
    x = x.reshape((-1, 1))
    errors = Jac @ x - Flow
    loss = np.linalg.norm(errors, ord=1)
    return loss


opt_res = minimize(cost, np.zeros(6), args=(
    Jacobian, flow.reshape((-1, 1))))
np.linalg.norm(opt_res.x - np.array(MOTION))

# %%

# computed from jacobian
px_flows = np.matmul(Jacobian, MOTION)
px_flows = px_flows.reshape((-1, 2))

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
None