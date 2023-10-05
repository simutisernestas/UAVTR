# %%
from kornia.feature import LoFTR
import torch
import time

# %%
img1 = torch.rand(1, 1, 100, 100).cuda()
img2 = torch.rand(1, 1, 100, 100).cuda()
input = {"image0": img1, "image1": img1}
loftr = LoFTR('outdoor').cuda()

# %%
start = time.time()
out = None
for _ in range(100):
    out = loftr(input)
end = time.time()
(end - start)/100*1000 , "ms"

# %%
loftr

# %%
out.keys()

# %%
(out["keypoints0"] - out["keypoints1"]).mean(axis=0)

# sudo ifconfig enp1s0 -multicast