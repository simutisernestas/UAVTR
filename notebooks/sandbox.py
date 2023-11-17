# %%
import numpy as np

# %%

# 45 degree pitch around y
R = np.array([[0.707, 0, 0.707],
              [0, 1, 0],
              [-0.707, 0, 0.707]])

v_cam = np.array([0, 1, 0])
w_cam = np.array([0, 0, 1])

v_body = R @ v_cam
w_body = R @ w_cam

r = np.array([.1, 0, 0])
v_corrected = v_body + np.cross(w_body, -r)

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

K[:2,:2] @ np.ones((2,6))
