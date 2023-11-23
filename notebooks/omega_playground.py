# %%
import numpy as np
import transforms3d as tf
# %%

def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])


dz = .01
dx = .02
dy = .03
q1 = tf.euler.euler2quat(0, 0, 0, 'sxyz')
q2 = tf.euler.euler2quat(dx, dy, dz, 'sxyz')
dt = 1

w = angular_velocities(q1, q2, dt)
print(w, dx/dt, dy/dt, dz/dt)

dq = tf.quaternions.qmult(q2, tf.quaternions.qinverse(q1))
euler = tf.euler.quat2euler(dq, 'sxyz')
w2 = np.array(euler) / dt

w, w2

# %%%

T = np.array([[-0.69055, -0.476984, 0.544549, 0.0451919],
              [-0.723336, 0.475584, -0.500685, -0.123328],
              [-0.0203437, -0.739457, -0.67351, -0.0670433],
              [0, 0, 0, 1]])

R = T[:3, :3]
t = T[:3, 3]

v_cam = np.ones(3)  # np.random.randn(3)
w_cam = np.ones(3)  # np.random.randn(3)
twist = np.concatenate([v_cam, w_cam])

v_body = R @ v_cam
w_body = R @ w_cam
v_corrected = v_body - np.cross(w_body, t)

def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


adjT = np.block([[R, skew_symmetric(t) @ R],
                 [np.zeros((3, 3)), R]])

np.allclose((adjT @ twist)[:3], v_corrected), v_body, v_corrected

