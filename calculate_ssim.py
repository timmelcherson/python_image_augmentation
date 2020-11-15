import os
import numpy as np
import cv2
import glob
import json
import math
from skimage.metrics import structural_similarity as ssim


def calculate_all_ssim(image_src):

    file_list = sorted(glob.glob(image_src + "*.jpg"))

    og_dict = {}
    aug_dict = {}

    for file_path in file_list:

        filename = (file_path.split('\\')[1]).split('.')[0]

        if "_" in filename:
            aug_dict[filename] = file_path
        else:
            og_dict[filename] = file_path

    ssim_dict = {}

    for i, key in enumerate(og_dict.keys()):

        original_image = cv2.imread(og_dict.get(key))
        augmented_score = {}

        for j, nested_key in enumerate(aug_dict.keys()):
            if key in nested_key:

                augmented_image = cv2.imread(aug_dict.get(nested_key))
                augmented_score[nested_key] = ssim(
                    original_image, augmented_image, multichannel=True)

        ssim_dict[key] = augmented_score
        print(str(len(og_dict) - i - 1) + " files left")

    with open('ssim_result.json', 'w') as fp:
        json.dump(ssim_dict, fp,  indent=4)

    return ssim_dict


def main():

    script_dir = os.path.dirname(__file__)
    image_src = os.path.join(script_dir, './augmented_test_images3/')

    ssim = calculate_all_ssim(image_src)

    print(ssim)


if __name__ == "__main__":
    main()
