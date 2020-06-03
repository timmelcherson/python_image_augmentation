import os
import shutil
import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

def plot_histograms_for_image(img, **kwargs):

    # histogram = cv2.calcHist([cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)],[0],None,[256],[0,256])
    fig = plt.figure(figsize=(14,8),constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    f2_ax1 = fig.add_subplot(spec2[0, 0])
    f2_ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    f2_ax1.set_xlabel('image width')
    f2_ax1.set_ylabel('image height')

    f2_ax2 = fig.add_subplot(spec2[0, 1])
    f2_ax2.hist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).ravel(),256,[0,256]) 
    f2_ax2.set_xlabel('pixel value')
    f2_ax2.set_ylabel('count')

    for ar in kwargs:
        if ar == 'gamma':
            fig.suptitle("Gamma = {}".format(kwargs.get('gamma')))

    print(type(fig))
    return fig


def main():
    
    src = 'images2'
    dst = 'images_dst'
    gamma_values = [0.3, 0.7, 1.3, 1.7]
    images = []
    gamma_histograms = []

    img = cv2.imread('images/pbear1.jpg')

    # fig1 = plot_histograms_for_image(img, gamma=1.0)
    # plt.show(fig1)
    # cv2.imshow("original image", img)
    # cv2.imshow("brighter image", brighter_img)
    # cv2.waitKey(0)


if __name__ == "__main__":
    main()