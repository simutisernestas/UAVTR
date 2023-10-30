import os
import time

import onnx
import onnxruntime
import torch
from PIL import Image

from detr_demo import detect, detr, transform

# Directory path containing the images
image_dir = 'images'

# Get a list of image filenames in the directory
image_filenames = sorted(
    [filename for filename in os.listdir(image_dir) if filename.endswith('.jpg')])

file = image_filenames[0]
print(file)

# for file in image_filenames[200:]:
# Read the next image
image_path = os.path.join(image_dir, file)
im = Image.open(image_path, 'r').convert('RGB')
print("aspect ratio:", im.size[0] / im.size[1])
# reduce aspect ratio
# if im.size[0]/im.size[1] > 1.5:
#     im = im.resize((int(im.size[0]/1.5), int(im.size[1]/1.1)))
print("aspect ratio:", im.size[0] / im.size[1])
print(im.size)

detr.eval()
start_time = time.time()
# for i in range(10):
scores, boxes = detect(im, detr, transform)
end_time = time.time()
print("Time taken: ", end_time - start_time)
print(scores.shape)

img = transform(im).unsqueeze(0)
print(img.shape)

print(torch.mean(img))
print(img[0, :, :10, 0])

# MUST USE TORCH NIGHTLY BUILD :)

# model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

# # Export the model
# torch.onnx.export(detr,
#                   img,
#                   "detr.onnx",
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=16,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names=['input'],   # the model's input names
#                   output_names=['logits', 'boxes'],)  # the model's output names

# load onnx model

onnx_model = onnx.load("detr.onnx")
onnx.checker.check_model(onnx_model)

# do inference with onnxruntime
ort_session = onnxruntime.InferenceSession("detr.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0].shape, ort_outs[1].shape)


def softmax(x):
    maxes = torch.max(x, -1, keepdim=True)[0]
    print(maxes.shape)
    x_exp = torch.exp(x - maxes)
    print(x_exp.shape)
    x_exp_sum = torch.sum(x_exp, -1, keepdim=True)
    print(x_exp_sum.shape)
    return x_exp / x_exp_sum


probas = torch.tensor(ort_outs[0]).softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.7
onnx_probs = probas[keep]

# use softmax functions defined above
scores_np = softmax(torch.tensor(ort_outs[0]))[0, :, :-1]
keep_np = scores_np.max(-1).values > 0.7
np_probs = scores_np[keep_np]

print(ort_outs[0][0, 0])

print(torch.allclose(onnx_probs, scores, rtol=1e-03, atol=1e-05))
print(torch.allclose(np_probs, scores, rtol=1e-03, atol=1e-05))
# np.allclose(np_probs, to_numpy(scores), rtol=1e-03, atol=1e-05)

# ort_outs[0] = ort_outs[0][keep]
# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(scores), ort_outs[0], rtol=1e-03, atol=1e-05)

print(ort_outs[1][0, keep])
