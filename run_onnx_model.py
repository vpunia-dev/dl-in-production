# Some standard imports
import io
import numpy as np
import onnx, onnxruntime
import torch
# preparing input image to test the onnx runtime 

from PIL import Image
import torchvision.transforms as transforms

onnx_model_path = "super_resolution.onnx"  
input_sample_path = "./data/cat.jpg"
output_sample_path = "./data/cat_superres_with_ort.jpg"
# create inference session from model file
ort_session = onnxruntime.InferenceSession(onnx_model_path)


def preprocess_input_sample(input_path):
    # replace this functions
    img = Image.open(input_path)
    resize = transforms.Resize([224, 224])
    img = resize(img)
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)
    return img_y, img_cb, img_cr


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


img_y, img_cb, img_cr = preprocess_input_sample(input_sample_path)

# get the inference results from model
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

def post_process_img(img_out_y):
    # replace this function
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")
    return final_img

final_img = post_process_img(img_out_y)

# Save the image, we will compare this with the output image from mobile device
final_img.save(output_sample_path)
