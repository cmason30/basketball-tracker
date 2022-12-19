#! /usr/bin/python3

from infer_overlay import load_model, run_inference, overlay_img
import cv2
import os
import re



def generate_image_seq(dir_path):
    list_dir = os.listdir(dir_path)
    len_dir = len(list_dir)

    cnt_ = 0

    for file in list_dir:
        if not file.endswith('.DS_Store'):

            num = re.findall(r'\d+', file)[0]
            abs_pth = f'{dir_path}/{file}'
            out_name = f'{num}_skeleton.png'

            out_dir = r'/Users/colinmason/Desktop/python-projects/basketball-tracker/data/cuts/frames_overlaid/'
            isExist = os.path.exists(f'{out_dir}/{out_name}')
            if isExist:
                cnt_ +=1
                continue

            model = load_model()

            output, image = run_inference(model, abs_pth)
            overlaid_img = overlay_img(model, output, image, 0.25, 0.65)

            cv2.imwrite(f"/Users/colinmason/Desktop/python-projects/basketball-tracker/data/cuts/frames_overlaid/{out_name}", overlaid_img)


            cnt_ += 1
            print(f"Saved image {cnt_}/{len_dir}")


# def make_png(cv2_array, filename):
#     cv2.write(filename, cv2_array)


generate_image_seq(r'/Users/colinmason/Desktop/python-projects/basketball-tracker/data/cuts/frames')