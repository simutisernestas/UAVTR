# %%
import matplotlib.pyplot as plt
from kornia.feature import LoFTR
import time
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
# %%

paths = os.listdir("/home/ernie/thesis/ros_ws/images/")
paths.sort()


# set CUDA_VISIBLE_DEVICES to 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%

# /home/ernie/thesis/ros_ws/images
# read images from this dir named 1.jpg, 2.jpg, ...
images = [Image.open(
    f"/home/ernie/thesis/ros_ws/images/{i}").convert('L') for i in paths if i != 'imu.txt']
images[50]

# %%
# read imu data
imu_data = np.loadtxt(
    "/home/ernie/thesis/ros_ws/images/imu.txt")
# timestamp, ax, ay, az
imu_data


image_timestamps = [float(".".join(i.split(".")[:2]))
                    for i in paths if i != 'imu.txt']
image_timestamps, imu_data[0, 0]

# integrate thoes between images
integrated = []
t = np.zeros((3,))
v = np.zeros((3,))
for idx in range(1, len(imu_data)):
    dt = imu_data[idx, 0] - imu_data[idx-1, 0]
    v += imu_data[idx, 1:]*dt
    t += v*dt + 0.5*imu_data[idx, 1:]*dt**2
    t_store = np.concatenate((imu_data[idx, 0:1], t))
    integrated.append(t_store.copy())
integrated = np.array(integrated)

image_translations = []
for img_t in range(1, len(image_timestamps)):
    t0 = image_timestamps[img_t-1]
    t1 = image_timestamps[img_t]

    # find imu data between t0 and t1
    integrated_between = integrated[(integrated[:, 0] >= t0) & (
        integrated[:, 0] <= t1)]

    if integrated_between.shape[0] == 0:
        image_translations.append(np.zeros((3,)))
        continue
    delta = integrated_between[-1, 1:] - integrated_between[0, 1:]
    image_translations.append(delta)

# %%
image_translations

# %%
images[0]

# %%
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((int(480/1.75), int(640/1.75))),
])

# %%
img1 = transform(images[0]).unsqueeze(0).float().cuda() / 255.
img2 = transform(images[50]).unsqueeze(0).float().cuda() / 255.
input = {"image0": img1, "image1": img1}
loftr = LoFTR('outdoor').cuda()
img1.shape

# %%
start = time.time()
out = None
out = loftr(input)
end = time.time()
(end - start)/100*1000, "ms"

# %%
out.keys(), out["confidence"].shape, out["keypoints0"], out["keypoints1"].shape

# filter keypoints with low confidence
kp0 = out["keypoints0"][out["confidence"] > 0.99]
kp1 = out["keypoints1"][out["confidence"] > 0.99]

K = np.eye(3)
# write the intrinsic matrix here
K[0, 0] = 385.4024658203125
K[0, 2] = 322.1327209472656
K[1, 1] = 384.882080078125
K[1, 2] = 240.0128173828125
K /= 1.75
E, mask = cv2.findEssentialMat(kp0.cpu().numpy(), kp1.cpu().numpy(), K)
retval, R, t, mask = cv2.recoverPose(E, kp0.cpu().numpy(), kp1.cpu().numpy())
s = 1.0
print(R, s*t)
# P0 = K @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
# P1 = K @ np.concatenate((R, t*s), axis=1)

# %%time

trajectory = []
R_vo = np.eye(3)
t_vo = np.zeros((3, 1))
# VO loop
for i in range(2, len(paths)//5):
    img1 = transform(images[i-1]).unsqueeze(0).float().cuda() / 255.
    img2 = transform(images[i]).unsqueeze(0).float().cuda() / 255.
    input = {"image0": img1, "image1": img2}
    out = loftr(input)
    kp0 = out["keypoints0"][out["confidence"] > 0.9]
    kp1 = out["keypoints1"][out["confidence"] > 0.9]
    E, mask = cv2.findEssentialMat(kp0.cpu().numpy(), kp1.cpu().numpy(), K)
    retval, R, t, mask = cv2.recoverPose(
        E, kp0.cpu().numpy(), kp1.cpu().numpy())
    s = np.linalg.norm(image_translations[i-1])
    t_vo = t_vo + s*(R_vo @ t)
    R_vo = R @ R_vo
    trajectory.append(t_vo)
    # if i > 10:
    #     break
    # t_f = t_f + scale*(R_f*t);
    # R_f = R*R_f;

# %%

# plot 3d trajectory

trajectory = np.array(trajectory)
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
# mark start point
ax.scatter(trajectory[0, 0], trajectory[0, 1],
           trajectory[0, 2], c='r', marker='o')
# end
ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
           trajectory[-1, 2], c='g', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


# %%

# %%
len(image_translations), len(paths)
