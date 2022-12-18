import torch
from torchvision import transforms
import cv2
import numpy as np
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint, plot_skeleton_kpts
import matplotlib.pyplot as plt




def load_model():

    model = torch.load('/Users/colinmason/Desktop/python-projects/basketball-tracker/yolov7-w6-pose.pt', map_location=device)['model']

    model.float().eval()

    if torch.cuda.is_available():
        model.half()
    return model



def run_inference(url):
    image = cv2.imread(url)

    image = letterbox(image, 960, stride=64, auto=True)[0]

    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    output, _ = model(image)
    return output, image



def visualize_output(output, image):
    output = non_max_suppression_kpt(output,
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    plt.figure(figsize=(12, 12))
    plt.axis('off')

    cv2.imshow('img', nimg)
    cv2.waitKey(0)


if __name__ == "__main__":
    model = load_model()

    output, image = run_inference(r'/Users/colinmason/Desktop/python-projects/basketball-tracker/data/cuts/out1.png')
    visualize_output(output, image)

