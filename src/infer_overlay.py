import torch
from torchvision import transforms
import cv2
import numpy as np
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint, plot_skeleton_kpts
import matplotlib.pyplot as plt




def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load('/Users/colinmason/Desktop/python-projects/basketball-tracker/yolov7-w6-pose.pt', map_location=device)['model']

    model.float().eval()

    if torch.cuda.is_available():
        model.half()
    return model



def run_inference(model, url):
    image = cv2.imread(url)

    image = letterbox(image, 960, stride=64, auto=True)[0]

    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    output, _ = model(image)
    return output, image



def overlay_img(model, output, image, ci, iou):
    output = non_max_suppression_kpt(output,ci,iou,nc=model.yaml['nc'],nkpt=model.yaml['nkpt'],kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

    nout_img = image[0].permute(1, 2, 0) * 255
    nout_img = nout_img.numpy().astype(np.uint8)
    nout_img = cv2.cvtColor(nout_img, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nout_img, output[idx, 7:].T, 3) # draws the skeleton on image

    nout_img = cv2.cvtColor(nout_img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('img', nout_img)
    # cv2.waitKey(0)
    #cv2.imwrite("test1.png", nout_img)

    return nout_img


if __name__ == "__main__":
    model = load_model()

    output, image = run_inference(model, r'/Users/colinmason/Desktop/python-projects/basketball-tracker/data/cuts/frames/out1.png')
    overlay_img(model, output, image, 0.25, 0.65)

