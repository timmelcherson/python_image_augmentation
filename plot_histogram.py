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
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
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
            fig.suptitle("Gamma = {}".format(kwargs.get('gamma')))

    print(type(fig))
    return fig


def plot_rbg_histograms(img, **kwargs):

    color = ('b', 'g', 'r')

    for channel, col in enumerate(color):
        histr = cv2.calcHist([img], [channel], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    
    for ar in kwargs:
        if ar == 'gamma':
            plt.title("Gamma = {}".format(kwargs.get('gamma')))
        else:
            plt.title('Histogram for color scale picture')
    
    plt.show()


def main():

    src = r'C:\Users\A560655\Documents\python\python_image_augmentation\images_dst'
    dst = 'images_dst'

    # img1 = cv2.imread('images_dst/bird1.jpg')
    # img2 = cv2.imread('images_dst/bird1_ga_03.jpg')
    # img3 = cv2.imread('images_dst/bird1_ga_07.jpg')
    # img4 = cv2.imread('images_dst/bird1_ga_20.jpg')
    # img5 = cv2.imread('images_dst/bird1_ga_30.jpg')

    gamma_values = [1.0, 0.3, 0.7, 2.0, 3.0]
    for index, item in enumerate(os.listdir(src)):
        filename = os.path.join(src, item)
        img = cv2.imread(filename)
        fig = plot_histograms_for_image(img, gamma=gamma_values[index])
        plt.show(fig)

    # fig1 = plot_histograms_for_image(img1)
    # fig2 = plot_histograms_for_image(img2, gamma=0.3)
    # fig3 = plot_histograms_for_image(img3, gamma=0.7)
    # fig4 = plot_histograms_for_image(img4, gamma=2.0)
    # fig5 = plot_histograms_for_image(img5, gamma=3.0)

    # fig1 = plot_rbg_histograms(img1)
    # fig2 = plot_rbg_histograms(img2, gamma=0.3)
    # fig3 = plot_rbg_histograms(img3, gamma=0.7)
    # fig4 = plot_rbg_histograms(img4, gamma=2.0)
    # fig5 = plot_rbg_histograms(img5, gamma=3.0)

    # fig1 = plot_histograms_for_image(img, gamma=1.0)
    # plt.show(fig1)
    # plt.show(fig2)
    # plt.show(fig3)
    # plt.show(fig4)
    # plt.show(fig5)
    # cv2.imshow("original image", img)
    # cv2.imshow("brighter image", brighter_img)
    # cv2.waitKey(0)


if __name__ == "__main__":
    main()
