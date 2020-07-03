import os
import shutil
import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


### This function will plot histograms for a grayscale images (with or without noise)
def plot_histograms_for_image(img, **kwargs):

    # histogram = cv2.calcHist([cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)],[0],None,[256],[0,256])
    fig = plt.figure(figsize=(14, 6))
    spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    f2_ax1 = fig.add_subplot(spec2[0, 0])
    f2_ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    f2_ax1.set_xlabel('image width')
    f2_ax1.set_ylabel('image height')

    f2_ax2 = fig.add_subplot(spec2[0, 1])
    f2_ax2.hist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).ravel(), 256, [0, 256])
    f2_ax2.set_xlabel('pixel value')
    f2_ax2.set_ylabel('count')

    for ar in kwargs:
        if ar == 'gamma':
            fig.suptitle("Image with Gamma {}".format(kwargs.get('gamma')), fontsize=18)

        elif ar == 'variance':
            fig.suptitle("Image with Added Noise with Variance {}".format(kwargs.get('variance')), fontsize=18)
            
    plt.show()


## This function will plot histograms for a coloured image (BGR-format since cv2 uses this format as standard)
def plot_rbg_histograms(img, **kwargs):

    # color = ('b', 'g', 'r')

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for channel, col in enumerate(color):
    #     # histr = cv2.calcHist([img], [channel], None, [256], [0, 256])
    #     # ax.hist(cv2.cvtColor([img], [channel],
    #     #                      cv2.COLOR_BGR2GRAY).ravel(), color=col, 256, [0, 256])
    #     ax.xlim([0, 256])
    
    # for ar in kwargs:
    #     if ar == 'gamma':
    #         plt.title("Gamma = {}".format(kwargs.get('gamma')))
    #     else:
    #         plt.title('Histogram for color scale picture')

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    fig_sub1 = fig.add_subplot(spec[0, 0])
    fig_sub2 = fig.add_subplot(spec[0, 1])

    for ar in kwargs:

        if ar == 'type':
            
            if kwargs.get('type') == 'original':
                fig_sub1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
                fig_sub2.hist(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).ravel(), 256, [0, 256])
                fig.suptitle(
                    "Original Data Set Image", fontsize=18)
     
            elif kwargs.get('type') == 'grey':
                fig_sub1.imshow(img, cmap='gray')
                fig_sub2.hist(img.ravel(), 256, [0, 256])
                fig.suptitle(
                    "Grayscale Image", fontsize=18)
        
    fig_sub1.set_xlabel('image width')
    fig_sub1.set_ylabel('image height')
    fig_sub2.set_xlabel('pixel value')
    fig_sub2.set_ylabel('count')
    
    plt.show()

    # fig1 = plt.figure(figsize=(9, 3), constrained_layout=True)
    # spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)
    # fig1_sub1 = fig1.add_subplot(spec1[0, 0])
    # fig1_sub1.imshow(grayImage, cmap='gray')
    # fig1_sub1.set_xlabel('image width')
    # fig1_sub1.set_ylabel('image height')

    # fig1_sub2 = fig1.add_subplot(spec1[0, 1])
    # fig1_sub2.hist(grayImage.ravel(), 256, [0, 256])
    # fig1_sub2.set_xlabel('pixel value')
    # fig1_sub2.set_ylabel('count')
    
    
    
    
    plt.show()


def main():

    original_src = r'C:\Users\A560655\Documents\python\python_image_augmentation\original_images'
    grey_src = r'C:\Users\A560655\Documents\python\python_image_augmentation\grey_images'
    gamma_src = r'C:\Users\A560655\Documents\python\python_image_augmentation\gamma_images'
    grey_noise_src = r'C:\Users\A560655\Documents\python\python_image_augmentation\grey_noise_images'

    gamma_values = [0.3, 0.7, 2.0, 3.0]
    variances = [0.1, 0.2, 0.3]

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

    # Plot histogram for any gamma adjusted image placed in the folder gamma_images
    for index, item in enumerate(os.listdir(gamma_src)):
        filename = os.path.join(gamma_src, item)
        img = cv2.imread(filename)
        plot_histograms_for_image(img, gamma=gamma_values[index])
    
    # # Plot histogram for any grayscale image with added noise placed in the folder grey_noise_images
    # for index, item in enumerate(os.listdir(grey_noise_src)):
    #     filename = os.path.join(grey_noise_src, item)
    #     img = cv2.imread(filename)
    #     plot_histograms_for_image(img, variance=variances[index])

    

    
        


if __name__ == "__main__":
    main()
