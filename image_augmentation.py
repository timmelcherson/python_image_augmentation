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


# Image converter progression counter
GLOBAL_COUNTER = count()

### Converts images to grayscale
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


### Converts images to grayscale and applies random noise
def convert_to_grayscale_with_noise(src, dst):
    
    standard_devs = [0.1, 0.2, 0.3]

    for item in os.listdir(src):

        s = os.path.join(src, item)
        split = item.split('.')

        if s.endswith(".jpg"): 
            originalImage = cv2.imread(s)
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            im_arr = np.asarray(grayImage)
            for i in range(len(standard_devs)):
                standard_dev = standard_devs[i]
                noise_img = random_noise(im_arr, mode='gaussian', var=standard_dev**2)
                test_arr = noise_img.flatten()
                noise_img = (255*noise_img).astype(np.uint8)

                var_split = str(round(standard_dev, 1)).split('.')
                jpg_filename = str(split[0] + "_gn_" + var_split[0] + var_split[1] + "." + split[1])
                cv2.imwrite(os.path.join(dst, jpg_filename), noise_img)
                file_counter()

        elif s.endswith(".txt"):
            for i in range(len(standard_devs)):
                standard_dev = standard_devs[i]
                var_split = str(round(standard_dev, 1)).split('.')
                txt_filename = str(split[0] + "_gn_" + var_split[0] + var_split[1] + "." + split[1])
                shutil.copy(s, os.path.join(dst, txt_filename))
                file_counter()

        else:
            print("Neither .jpg file or .txt file")

        


### Adjust gamma values of images
def adjust_gamma(src, dst):
    
    gamma_values = [0.3, 0.7, 2.0, 3.0]

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


### Copy files without augmentation
def copy_original_files(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        shutil.copy(s, dst)
        file_counter()

def file_counter():
    print("{} files completed.".format(next(GLOBAL_COUNTER)))
    

### Image augmentation script
### The augmented images are currently saved in separate folders
### To save all augmented images from a data set to a specifing
### training or test set, change the relative sources correspondingly
def main():
    
    script_dir = os.path.dirname(__file__)

    # data_src = './any_data_set_source_folder/'
    # data_dst = './any_data_destination_folder/'
    relative_src = './original_images/' # Or training/test set folders
    relative_grayscale_dst = './images_dst/' # Or training/test set folders
    relative_grayscale_noise_dst = './images_dst/' # Or training/test set folders
    relative_gamma_dst = './images_dst/' # Or training/test set folders

    src = os.path.join(script_dir, relative_src)
    grayscale_dst = os.path.join(script_dir, relative_grayscale_dst)
    grayscale_noise_dst = os.path.join(script_dir, relative_grayscale_noise_dst)
    gamma_dst = os.path.join(script_dir, relative_gamma_dst)
    
    convert_to_grayscale(src, grayscale_dst)
    convert_to_grayscale_with_noise(src, grayscale_noise_dst)
    adjust_gamma(src, gamma_dst)

    # Uncomment to copy original files to eventual training / test set folder
    # copy_original_files(data_src, dst)
    

if __name__ == "__main__":
    main()
