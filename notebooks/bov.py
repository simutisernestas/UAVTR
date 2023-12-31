# %%
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2
import os

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
