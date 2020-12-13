import os
import shutil
import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


### Plot histograms for a grayscale images (with or without noise)
def plot_histograms_for_image(img, **kwargs):

    fig = plt.figure(figsize=(14, 6))
    spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    f2_ax1 = fig.add_subplot(spec2[0, 0])
    f2_ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    f2_ax1.set_xlabel('Image width')
    f2_ax1.set_ylabel('Image height')

    f2_ax2 = fig.add_subplot(spec2[0, 1])
    f2_ax2.hist(img.ravel(), 256, [0, 256])
    f2_ax2.set_xlabel('Pixel value')
    f2_ax2.set_ylabel('Amount of pixels')

    for ar in kwargs:
        if ar == 'gamma':
            fig.suptitle("Image with Gamma {}"
              .format(kwargs.get('gamma')),fontsize=18)

        elif ar == 'std_dev':
            fig.suptitle("Image with Added Noise with Standard Deviation {}"
              .format(kwargs.get('std_dev')), fontsize=18)
            
    plt.show()


## Plot histograms for a coloured image 
# (BGR-format since cv2 uses this format as standard)
def plot_rbg_histograms(img, **kwargs):

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    fig_sub1 = fig.add_subplot(spec[0, 0])
    fig_sub2 = fig.add_subplot(spec[0, 1])
    fig_sub3 = fig.add_subplot(spec[1, 0])
    fig_sub4 = fig.add_subplot(spec[1, 1])
    fig_sub1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')

    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    print(blue.shape)

    colors = ("Blue", "Green", "Red")

    # for blue_value, green_value, red_value in zip(blue, green, red):

    fig_sub2.hist(blue.ravel(), 256, [0, 256], color="b")
    fig_sub3.hist(green.ravel(), 256, [0, 256], color="g")
    fig_sub4.hist(red.ravel(), 256, [0, 256], color="r")

    fig_sub1.set_title('Original image')
    fig_sub2.set_xlabel('Image width')
    fig_sub2.set_ylabel('Image height')

    fig_sub2.set_title('Blue channel')
    fig_sub2.set_xlabel('Pixel value')
    fig_sub2.set_ylabel('Amount of pixels')

    fig_sub3.set_title('Green channel')
    fig_sub3.set_xlabel('Pixel value')
    fig_sub3.set_ylabel('Amount of pixels')

    fig_sub4.set_title('Red channel')
    fig_sub4.set_xlabel('Pixel value')
    fig_sub4.set_ylabel('Amount of pixels')

    fig.suptitle('An Image and its RGB Colour Channels', fontsize=20)
    plt.show()

    fig2 = plt.figure(figsize=(10, 6))
    fig2_sub = fig2.add_subplot()

    fig2_sub.hist(blue.ravel(), 256, [0, 256], color="b")
    fig2_sub.hist(green.ravel(), 256, [0, 256], color="g")
    fig2_sub.hist(red.ravel(), 256, [0, 256], color="r")

    fig2.suptitle('Combined RGB Colour channels', fontsize=20)
    fig2_sub.set_xlabel('Pixel value')
    fig2_sub.set_ylabel('Amount of pixels')

    plt.show()
    

def main():

    script_dir = os.path.dirname(__file__)

    original_relative_path = './original_images/'
    grey_relative_path = './grey_images/'
    gamma_relative_path = './gamma_images/'
    grey_noise_relative_path = './grey_noise_images/'

    original_src = os.path.join(script_dir, original_relative_path)
    grey_src = os.path.join(script_dir, grey_relative_path)
    gamma_src = os.path.join(script_dir, gamma_relative_path)
    grey_noise_src = os.path.join(script_dir, grey_noise_relative_path)

    gamma_values = [0.3, 0.7, 2.0, 3.0]
    std_devs = [0.1, 0.2, 0.3]

    # # Plot histogram for any non-augmented image placed in the folder original_images
    # for index, item in enumerate(os.listdir(original_src)):
    #     filename = os.path.join(original_src, item)
    #     img = cv2.imread(filename)
    #     plot_rbg_histograms(img, type='original')
        

    # # Plot histogram for any grayscale image placed in the folder grey_images
    # for index, item in enumerate(os.listdir(grey_src)):
    #     filename = os.path.join(grey_src, item)
    #     img = cv2.imread(filename)
    #     plot_rbg_histograms(img, type='grey')

    # # Plot histogram for any gamma adjusted image placed in the folder gamma_images
    # for index, item in enumerate(os.listdir(gamma_src)):
    #     filename = os.path.join(gamma_src, item)
    #     img = cv2.imread(filename)
    #     plot_histograms_for_image(img, gamma=gamma_values[index])
    
    # Plot histogram for any grayscale image with added noise placed in the folder grey_noise_images
    for index, item in enumerate(os.listdir(grey_noise_src)):
        filename = os.path.join(grey_noise_src, item)
        img = cv2.imread(filename)
        plot_histograms_for_image(img, std_dev=std_devs[index])


if __name__ == "__main__":
    main()
