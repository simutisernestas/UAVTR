# %%
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
# nice print of numpy array
np.set_printoptions(precision=5, suppress=True)


def e2h(e):
    if e.ndim == 1:
        return np.append(e, 1)

    return np.vstack((e, np.ones(e.shape[1])))


def h2e(h):
    if h.ndim == 1:
        return h[:-1]/h[-1]

    return h[:-1, :]/h[-1, :]


def project(P, C, noise=0):
    e = np.random.normal(0, noise, (2, P.shape[1]))
    return h2e(C @ e2h(P)) + e


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

# %%


u0 = 640//2    # principal point, horizontal coordinate
v0 = 480//2    # principal point, vertical coordinate
K = np.array([[285, 0, u0],
              [0, 285, v0],
              [0, 0, 1]])
Kinv = np.linalg.inv(K)
P0 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]])
vel = 1
dt = 1/10
dx = vel * dt
dy = vel * dt
dz = vel * dt
P1 = np.array([[1, 0, 0, dx],
               [0, 1, 0, dy],
               [0, 0, 1, dz]])
X = np.identity(4)
C0 = K @ P0 @ X
C1 = K @ P1 @ X


def get_virtual_BEV_camera(R, t):
    newP = P0.copy()
    newP[:3, :3] = R
    newP[:3, 3] = t
    return K @ newP @ X

# %%


Rx = np.array([[1, 0, 0],
               [0, 0, -1],
               [0, 1, 0]])
Ry = np.array([[0, 0, 1],
               [0, 1, 0],
               [-1, 0, 0]])
CV = get_virtual_BEV_camera(Rx, np.array([0, 5, -5]))

Points = np.array([
    [0, 0, 4],
    [0, 0, 5],
    [0, 0, 6],
])

NOISE = 1e-6
projected0 = project(Points.T, CV, NOISE)

# Plotting remains the same
plt.scatter(projected0.T[:, 0], projected0.T[:, 1], c='r')
plt.xlim(0, u0 * 2)
plt.ylim(0, v0 * 2)
plt.gca().invert_yaxis()
plt.show()


# %%

def getT(R, t):
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


cam2world = pt.transform_from_pq([0, 0, 0, np.sqrt(0.5), -np.sqrt(0.5), 0, 0])
# default parameters of a camera in Blender
sensor_size = np.array([0.036, 0.024])
intrinsic_matrix = np.array([
    [0.05, 0, sensor_size[0] / 2.0],
    [0, 0.05, sensor_size[1] / 2.0],
    [0, 0, 1]
])
virtual_image_distance = .2

ax = pt.plot_transform(A2B=cam2world, s=0.2)
pc.plot_camera(
    ax, cam2world=cam2world, M=intrinsic_matrix, sensor_size=sensor_size,
    virtual_image_distance=virtual_image_distance)

T = getT(Rx, np.array([0, .5, -.5]))
cam2world = T @ cam2world

pt.plot_transform(A2B=cam2world, s=0.2)
pc.plot_camera(
    ax, cam2world=cam2world, M=intrinsic_matrix, sensor_size=sensor_size,
    virtual_image_distance=virtual_image_distance)
plt.show()

# %%


def simulation(plot=False, depth_assumption=False,
               lower_bound=5, known_vz=False):
    np.random.seed(0)
    # Existing code for Points initialization
    Points = np.random.uniform(-4, 4, (3, 3))
    Points[:, 2] = np.random.uniform(
        lower_bound, lower_bound+3, Points.shape[0])
    print(Points)
    print(K)

    NOISE = 10 / lower_bound * 0
    # Vectorize projection calculations
    projected0 = project(Points.T, C0, NOISE)
    projected0_xy = Kinv @ e2h(projected0)
    projected1 = project(Points.T, C1, NOISE)
    projected1_xy = Kinv @ e2h(projected1)

    print(projected1.T)

    # Vectorize flow calculations
    flows = projected1 - projected0
    flows_xy = (projected0_xy[:2, :] - projected1_xy[:2, :]).T
    print(flows / dt)

    # Plotting remains the same
    if plot:
        plt.scatter(projected0.T[:, 0], projected0.T[:, 1], c='r')
        plt.scatter(projected1.T[:, 0], projected1.T[:, 1], c='g')
        plt.quiver(projected0.T[:, 0], projected0.T[:, 1], flows.T[:, 0], flows.T[:, 1],
                   color='b', label='flow', angles='xy', scale_units='xy', scale=1)
        plt.xlim(0, u0 * 2)
        plt.ylim(0, v0 * 2)
        plt.gca().invert_yaxis()
        plt.show()

    # Depth assumption
    if depth_assumption:
        Points[:, 2] = lower_bound

    Lx1 = Lx(projected1_xy.T, Points[:, 2])
    Lx1 = np.vstack(Lx1)
    for r in range(Lx1.shape[0]):
        if r % 2 == 0:
            Lx1[r, :] *= K[0, 0]
        else:
            Lx1[r, :] *= K[1, 1]
    print(Lx1)

    if known_vz:
        # assume that vz is knows, subract it from the flow
        dvx_flow = Lx1[:, 2] * -vel
        flows_xy = flows_xy.reshape(-1, 1) / dt
        flows_xy -= dvx_flow.reshape(-1, 1)
        x = np.linalg.pinv(Lx1[:, :2]) @ flows_xy
        return x

    flows_xy = flows_xy.reshape(-1, 1) / dt
    x = np.linalg.pinv(Lx1[:, :3]) @ flows_xy
    return x


def calc_error(res):
    gt = np.array([vel]*3 + [0]*3).reshape(-1, 1)
    if res.shape[0] == 3:
        res = np.append(res, np.zeros((3, 1)), axis=0)
    if res.shape[0] == 2:
        res = np.append(res, np.ones((1, 1)), axis=0)
        res = np.append(res, np.zeros((3, 1)), axis=0)
    error = np.linalg.norm(res - gt, axis=1, ord=1)
    return error


SIZE = 1
errors = np.zeros((SIZE, 6))
for i in range(SIZE):
    res = simulation(plot=False, lower_bound=30,
                     depth_assumption=False, known_vz=False)
    errors[i] = calc_error(res)
plt.figure(dpi=300)
plt.boxplot(errors[:, :3])
plt.xticks(np.arange(1, 4), ['vx', 'vy', 'vz'])
plt.ylabel('error (m/s)')
# plt.figure(dpi=300)
# plt.boxplot(errors[:, 3:])
# plt.xticks(np.arange(1, 4), ['wx', 'wy', 'wz'])
# plt.ylabel('error (rad/s)')

# %%

bound_errors = []
RANGE = range(5, 50, 5)
SIZE = 10
KNOWN_VZ = True
for bound in RANGE:
    errors = np.zeros((SIZE,))
    for i in range(SIZE):
        res = simulation(depth_assumption=False,
                         lower_bound=bound, known_vz=KNOWN_VZ)
        errors[i] = np.linalg.norm(calc_error(res), ord=1)

    errors2 = np.zeros((SIZE,))
    for i in range(SIZE):
        res = simulation(depth_assumption=True,
                         lower_bound=bound, known_vz=KNOWN_VZ)
        errors2[i] = np.linalg.norm(calc_error(res), ord=1)

    bound_errors.append([np.mean(errors), np.mean(errors2)])

# plot
plt.figure(dpi=300)
plt.scatter(RANGE, np.array(bound_errors)
            [:, 0], label='no depth assumption')
plt.scatter(RANGE, np.array(
    bound_errors)[:, 1], label='depth assumption')
plt.xlabel('lower bound of depth')
plt.ylabel('average error (m/s)')
plt.legend()
