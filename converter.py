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
from itertools import count


GLOBAL_COUNTER = count()
TOTAL_AMOUNT_OF_FILES = 1980 + 1980*1 + 1980*3 + 1980*4

### IMAGE CONVERTER TO GRAYSCALE
def convert_to_grayscale(src, dst):

    for item in os.listdir(src):

        s = os.path.join(src, item)
        split = item.split('.')

        if s.endswith(".jpg"): 
            originalImage = cv2.imread(s)
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            jpg_filename = str(split[0] + "_gr" + "." + split[1])
            cv2.imwrite(os.path.join(dst, jpg_filename), grayImage)
            file_counter()

        elif s.endswith(".txt"):
            txt_filename = split[0] + "_gr" + "." + split[1]
            shutil.copy(s, os.path.join(dst, txt_filename)) 
            file_counter()

        else:
            print("Neither .jpg file or .txt file")


### IMAGE CONVERTER TO GRAYSCALE WITH NOISE
### APPLIES 3 DIFFERENT VARIANCES 
def convert_to_grayscale_with_noise(src, dst):
    
    variances = [0.1, 0.2, 0.3]

    for item in os.listdir(src):

        s = os.path.join(src, item)
        split = item.split('.')

        if s.endswith(".jpg"): 
            originalImage = cv2.imread(s)
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            im_arr = np.asarray(grayImage)
            
            for i in range(len(variances)):
                variance = variances[i]
                noise_img = random_noise(im_arr, mode='gaussian', var=variance**2)
                noise_img = (255*noise_img).astype(np.uint8)

                var_split = str(round(variance, 1)).split('.')
                jpg_filename = str(split[0] + "_gn_" + var_split[0] + var_split[1] + "." + split[1])
                cv2.imwrite(os.path.join(dst, jpg_filename), noise_img)
                file_counter()

        elif s.endswith(".txt"):
            for i in range(len(variances)):
                variance = variances[i]
                var_split = str(round(variance, 1)).split('.')
                txt_filename = str(split[0] + "_gn_" + var_split[0] + var_split[1] + "." + split[1])
                shutil.copy(s, os.path.join(dst, txt_filename))
                file_counter()

        else:
            print("Neither .jpg file or .txt file")


### ADJUST GAMMA OF IMAGE
### APPLIES 4 DIFFERENT VALUES OF GAMMA
def adjust_gamma(src, dst):
    
    gamma_values = [0.3, 0.7, 1.3, 1.7]

    for item in os.listdir(src):

        s = os.path.join(src, item)
        split = item.split('.')

        if s.endswith(".jpg"): 
            originalImage = cv2.imread(s)
            
            for i in range(len(gamma_values)):
                # build a lookup table mapping the pixel values [0, 255] to
                # their adjusted gamma values
                gamma = gamma_values[i]
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")
                
                gamma_img = cv2.LUT(originalImage, table)

                gamma_split = str(round(gamma, 1)).split('.')
                jpg_filename = str(split[0] + "_ga_" + gamma_split[0] + gamma_split[1] + "." + split[1])
                cv2.imwrite(os.path.join(dst, jpg_filename), gamma_img)
                file_counter()

        elif s.endswith(".txt"):
            for i in range(len(gamma_values)):
                gamma = gamma_values[i]
                gamma_split = str(round(gamma, 1)).split('.')
                txt_filename = str(split[0] + "_ga_" + gamma_split[0] + gamma_split[1] + "." + split[1])
                shutil.copy(s, os.path.join(dst, txt_filename)) 
                file_counter()

        else:
            print("Neither .jpg file or .txt file")


### FUNCTION FOR SIMPLE COPYING OF FILES TO NEW DESTINATION
def copy_original_files(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        shutil.copy(s, dst)
        file_counter()

def file_counter():
    print("{} of {} files completed.".format(next(GLOBAL_COUNTER), TOTAL_AMOUNT_OF_FILES))
    

def main():
    
    src = r'D:\datasets\OIDv4_ToolKit\OID\Dataset\train\Bird_Polar bear'
    dst = r'D:\datasets\Bird_Polar bear'
    gamma_values = [0.3, 0.7, 1.3, 1.7]
    images = []
    gamma_histograms = []

    img = cv2.imread('images/pbear1.jpg')

    adjust_gamma(src, dst)
    convert_to_grayscale(src, dst)
    convert_to_grayscale_with_noise(src, dst)
    copy_original_files(src, dst)

    # for i in range(len(gamma_values)):
    #     gam_img = adjust_gamma(img, gamma_values[i])
    #     images.append(gam_img)
        # gamma_histograms.append(plot_histograms_for_image(gam_img, gamma=gamma_values[i]))
        

    # fig1 = plot_histograms_for_image(img, gamma=1.0)
    # plt.show(fig1)
    # cv2.imshow("original image", img)
    # cv2.imshow("brighter image", brighter_img)
    # cv2.waitKey(0)


if __name__ == "__main__":
    main()
