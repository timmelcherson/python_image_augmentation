import os
import shutil
import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage.util import random_noise


def convert_to_grayscale(src, dst):

    for item in os.listdir(src):

        s = os.path.join(src, item)
        split = item.split('.')

        if s.endswith(".jpg"): 
            originalImage = cv2.imread(s)
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            jpg_filename = str(split[0] + "_gr" + "." + split[1])
            cv2.imwrite(os.path.join(dst, jpg_filename), grayImage)

        elif s.endswith(".txt"):
            txt_filename = split[0] + "_gr" + "." + split[1]
            shutil.copy(s, os.path.join(dst, txt_filename)) 

        else:
            print("Neither .jpg file or .txt file")


# def adjust_exposure():


def convert_to_grayscale_with_noise(src, dst):
    
    for item in os.listdir(src):

        s = os.path.join(src, item)
        split = item.split('.')
        
        variance = 0.1

        if s.endswith(".jpg"): 
            originalImage = cv2.imread('images/bird1.jpg')
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            im_arr = np.asarray(grayImage)
            
            for i in range(3):
                noise_img = random_noise(im_arr, mode='gaussian', var=variance**2)
                noise_img = (255*noise_img).astype(np.uint8)

                var_split = str(round(variance, 1)).split('.')
                jpg_filename = str(split[0] + "_gn_" + var_split[0] + var_split[1] + "." + split[1])
                print(jpg_filename)
                cv2.imwrite(os.path.join(dst, jpg_filename), noise_img)
                variance += 0.2

        elif s.endswith(".txt"):
            for i in range(3):
                var_split = str(round(variance, 1)).split('.')
                txt_filename = str(split[0] + "_gn_" + var_split[0] + var_split[1] + "." + split[1])
                print(txt_filename)
                shutil.copy(s, os.path.join(dst, txt_filename)) 
                variance += 0.2

        else:
            print("Neither .jpg file or .txt file")

# def convert_to_grayscale():


# def copy_original_files:


# def navigate_and_rename(src):
#     jpgCounter = 0
#     txtCounter = 0

#     for item in os.listdir(src):

#         s = os.path.join(src, item)
#         split = item.split('.')

#         if s.endswith(".jpg"): 

#             jpg_filename = str(split[0] + str(jpgCounter) + "." + split[1])
#             shutil.copy(s, os.path.join(src, jpg_filename))   
#             jpgCounter += 1
#             print(jpg_filename)
#         elif s.endswith(".txt"):
#             txt_filename = split[0] + str(txtCounter) + "." + split[1]
#             shutil.copy(s, os.path.join(src, txt_filename)) 
#             txtCounter += 1
#             print(txt_filename)
#         else:
#             print("Neither .jpg file or .txt file")

         
def main():
    print("MAIN")
    src = 'images2'
    dst = 'images_dst'

    convert_to_grayscale(src, dst)
    convert_to_grayscale_with_noise(src, dst)


if __name__ == "__main__":
    main()
