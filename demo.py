import json
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def transform_image(infile):
    # demo only  will be modified for our specific purpose
    input_transforms = [transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)
    timg = my_transforms(image)
    timg.unsqueeze_(0)
    return timg


def get_prediction(input_tensor):
    # demo only  will be modified for our specific purpose
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction



def render_prediction(prediction_idx):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]

    return prediction_idx, class_name



#  this code too will be modified
img_class_map = None
mapping_file_path = 'index_to_name.json'                  # Human-readable names for Imagenet classes
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)

# main code here
# setting gpu/cpu based on cuda availability
# for multiple GPU we need to create an array, device names will be "cuda:i"
# where i will be [0,1...number of gpus-1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# we will use a different model but intialization will be same
# we are loading the model here
model = models.densenet121(pretrained=True).to(device)


# this code will run whenever worker deques a job from queue
in_img = Image.open('./kitten.jpg') # write the logic to read img data here
modified_img_tensor = transform_image(in_img).to(device)
# model is generating prediction here
prediction = get_prediction(modified_img_tensor)
class_id, class_name = render_prediction(prediction)

# printing the prediction
print(class_id, class_name)