# %%
import numpy.linalg as la
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
dt = 1/30
dx = vel * dt
dy = -vel * dt
dz = (vel-.5) * dt
vvec = np.array([dx, dy, dz]) / dt
P1 = np.array([[1, 0, 0, dx],
               [0, 1, 0, dy],
               [0, 0, 1, dz]])
w = .1
theta = w * dt
print(f"theta: {theta}, w: {w} rad/s")
Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]])
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta), np.cos(theta)]])
Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
               [0, 1, 0],
               [-np.sin(theta), 0, np.cos(theta)]])
# Once rotation is introduces it collapses : )
P1[:3, :3] = Rz @ P1[:3, :3]
P1[:3, :3] = Rx @ P1[:3, :3]
P1[:3, :3] = Ry @ P1[:3, :3]
X = np.identity(4)
C0 = K @ P0 @ X
C1 = K @ P1 @ X

# %%

def polygon_area(coords):
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    area = abs(area) / 2.0
    return area


def simulation(plot=False, depth_assumption=False, lower_bound=5):
    # Existing code for Points initialization
    Points = np.random.uniform(-10, 10, (4000, 3))
    # sort by y
    Points = Points[Points[:, 1].argsort()]
    Points[:, 2] = np.random.uniform(
        lower_bound, lower_bound+0, Points.shape[0])
    # for i in range(Points.shape[0]):
    #     Points[i, 2] += i * .005

    NOISE = 1 / lower_bound * 0
    projected0 = project(Points.T, C0, NOISE)
    projected0_xy = Kinv @ e2h(projected0)
    projected1 = project(Points.T, C1, NOISE)
    projected1_xy = Kinv @ e2h(projected1)

    flows = projected1 - projected0
    flows_xy = (projected0_xy[:2, :] - projected1_xy[:2, :]).T

    # flip sign of random flows; RANSAC should handle..
    # for i in range(flows.shape[1]):
    #     if np.random.rand() > .95:
    #         flows_xy[i] *= -1

    if plot:
        plt.scatter(projected0[0, :], projected0[1, :], c='r')
        plt.scatter(projected1[0, :], projected1[1, :], c='g')
        plt.quiver(projected0[0, :], projected0[1, :],
                   flows[0, :], flows[1, :],
                   color='b', label='flow',
                   angles='xy', scale_units='xy', scale=1)
        plt.xlim(0, u0 * 2)
        plt.ylim(0, v0 * 2)
        plt.gca().invert_yaxis()
        plt.show()

    if depth_assumption:
        Points[:, 2] = lower_bound

    Z = Points[:, 2]  # BEST! so the previous depth values go here?
    Lx1 = Lx(projected1_xy.T, Z)
    Lx1 = np.vstack(Lx1)

    flows_xy = flows_xy.reshape(-1, 1) / dt
    vel = np.linalg.pinv(Lx1) @ flows_xy
    return vel


SIZE = 10
gt = np.concatenate([vvec, np.ones(3)*w]).reshape(-1, 1)

print(gt.T)
errors = np.zeros((SIZE, 6))
for i in range(SIZE):
    res = simulation(lower_bound=15,
                     plot=False,
                     depth_assumption=False)
    if res.shape[0] == 3:
        res = np.append(res, np.zeros((3, 1)), axis=0)
    errors[i] = np.linalg.norm(res - gt, axis=1, ord=1)

# make statistics of error along every dimension
plt.figure(dpi=300)
plt.boxplot(errors[:, :3])
plt.xticks(np.arange(1, 4), ['vx', 'vy', 'vz'])
plt.ylabel('error (m/s)')
plt.figure(dpi=300)
plt.boxplot(errors[:, 3:])
plt.xticks(np.arange(1, 4), ['wx', 'wy', 'wz'])
plt.ylabel('error (rad/s)')

# %%

SIZE = 10
gt = np.array([vel]*3 + [0]*3).reshape(-1, 1)
errors = np.zeros((SIZE, 6))
for i in range(SIZE):
    res = simulation(depth_assumption=True)
    if res.shape[0] == 3:
        res = np.append(res, np.zeros((3, 1)), axis=0)
    errors[i] = np.linalg.norm(res - gt, axis=1, ord=1)

# make statistics of error along every dimension
plt.figure(dpi=300)
plt.boxplot(errors[:, :3])
plt.xticks(np.arange(1, 4), ['vx', 'vy', 'vz'])
plt.ylabel('error (m/s)')

# %%

bound_errors = []
RANGE = range(5, 50, 5)
for bound in RANGE:
    SIZE = 10
    gt = np.array([vel]*3 + [0]*3).reshape(-1, 1)
    errors = np.zeros((SIZE,))
    for i in range(SIZE):
        res = simulation(depth_assumption=False, lower_bound=bound)
        if res.shape[0] == 3:
            res = np.append(res, np.zeros((3, 1)), axis=0)
        errors[i] = np.linalg.norm(res - gt, ord=1)

    gt = np.array([vel]*3 + [0]*3).reshape(-1, 1)
    errors2 = np.zeros((SIZE,))
    for i in range(SIZE):
        res = simulation(depth_assumption=True, lower_bound=bound)
        if res.shape[0] == 3:
            res = np.append(res, np.zeros((3, 1)), axis=0)
        errors2[i] = np.linalg.norm(res - gt, ord=1)

    bound_errors.append([np.mean(errors), np.mean(errors2)])

# plot
plt.figure(dpi=300)
plt.plot(RANGE, np.array(bound_errors)
         [:, 0], label='no depth assumption')
plt.plot(RANGE, np.array(
    bound_errors)[:, 1], label='depth assumption')
plt.xlabel('lower bound of depth')
plt.ylabel('average error (m/s)')
plt.legend()


# %%

def skew(w):
    if isinstance(w, float):
        w = np.array([w, w, w])
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


lower_bound = 25
Points = np.random.uniform(-20, 20, (10000, 3))
Points[:, 2] = np.random.uniform(
    lower_bound, lower_bound+3, Points.shape[0])
for i in range(Points.shape[0]):
    Points[i, 2] += i * .005

NOISE = 1 / lower_bound * 0
projected0 = project(Points.T, C0, NOISE)
projected0_xy = Kinv @ e2h(projected0)
projected1 = project(Points.T, C1, NOISE)
projected1_xy = Kinv @ e2h(projected1)
Z = Points[:, 2]  # BEST! so the previous depth values go here?

vels = np.zeros((Points.shape[0], 3))
idxs = []
for i in range(Points.shape[0]):
    p0 = projected0_xy[:, i]
    p1 = projected1_xy[:, i]
    w = np.array([-.1, -.1, -.1]) + np.random.normal(0, .01, 3)
    skew_w = skew(w)
    v = ((Z[i] + dz) * (np.eye(3) + skew_w*dt) @ p1 - p0 * Z[i]) / dt
    vels[i] = v
    if np.linalg.norm(v - vvec) < 1e-1:
        idxs.append(i)

plt.plot(projected0[0, idxs], projected0[1, idxs], 'r.')
plt.ylim(0, 480)
plt.xlim(0, 640)

plt.figure()
plt.hist(vels[:, 0], bins=50, alpha=.5)
plt.hist(vels[:, 1], bins=50, alpha=.5)
plt.show()

# get most frequent float velocity value for each axis
vels = np.round(vels, 1)
vels = np.unique(vels, axis=0, return_counts=True)
vels[0][vels[1].argmax()]
