# %%
import numpy as np

# %%


#   -0.69055  -0.476984   0.544549  0.0451919
#  -0.723336   0.475584  -0.500685  -0.123328
# -0.0203437  -0.739457   -0.67351 -0.0670433
#          0          0          0          1
T = np.array([[-0.69055, -0.476984, 0.544549, 0.0451919],
              [-0.723336, 0.475584, -0.500685, -0.123328],
              [-0.0203437, -0.739457, -0.67351, -0.0670433],
              [0, 0, 0, 1]])
T

# %%

R = T[:3, :3]
t = T[:3, 3]
R, t


# %%

v_cam = np.random.randn(3)
w_cam = np.random.randn(3)
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

np.allclose((adjT @ twist)[:3], v_corrected)

# # print everything
# print("R:\n", R)
# print("v_cam:\n", v_cam)
# print("w_cam:\n", w_cam)
# print("v_body:\n", v_body)
# print("w_body:\n", w_body)
# print("v_corrected:\n", v_corrected)

# %%

f = 3
K = np.array([[f, 0, 0],
              [0, f, 0],
              [0, 0, 1]])

K[:2, :2] @ np.ones((2, 6))
