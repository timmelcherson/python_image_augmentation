# %%

####################################################################
#################### Adjust Exposure in Image ######################
####################################################################
from __future__ import print_function
from builtins import input
import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

image = cv2.imread('images/pbear1.jpg')

# image = Image.open('images/pbear1.jpg')

# im_arr = np.asarray(image)
# new_image = np.zeros(image.shape, image.dtype)

alpha = 1.0 # Simple contrast control
beta = 50.0    # Simple brightness control

new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

print("new image array: ")
print(new_image.shape)

# for i in range(683):
#     for j in range(1024):
#         for k in range(3):
#             if image[i][j][k] < abs(beta):
#                 print(new_image[i][j])
hist_before = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)],[0],None,[256],[0,256])
hist_after = cv2.calcHist([cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)],[0],None,[256],[0,256])


fig = plt.figure(figsize=(20,14),constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
f2_ax1 = fig.add_subplot(spec2[0, 0])
f2_ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
f2_ax2 = fig.add_subplot(spec2[0, 1])
f2_ax2.hist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).ravel(),256,[0,256]) 
f2_ax3 = fig.add_subplot(spec2[1, 0])
f2_ax3.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
f2_ax4 = fig.add_subplot(spec2[1, 1])
f2_ax4.hist(cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY).ravel(),256,[0,256]) 

# %%

####################################################################
################# Add Noise to Image v1 (working) ##################
####################################################################
import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage.util import random_noise

image = cv2.imread('images/pbear1.jpg')
# convert PIL Image to ndarray
im_arr = np.asarray(image)

# random_noise() method will convert image in [0, 255] to [0, 1.0],
# inherently it use np.random.normal() to create normal distribution
# and adds the generated noised back to image
noise_img = random_noise(im_arr, mode='gaussian', var=0.2**2)
noise_img = (255*noise_img).astype(np.uint8)


# img = Image.fromarray(noise_img)
# img.save("images/random_noise_image.jpg")
# img.show()

fig = plt.figure(figsize=(20,14),constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
f2_ax1 = fig.add_subplot(spec2[0, 0])
f2_ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
f2_ax2 = fig.add_subplot(spec2[0, 1])
f2_ax2.hist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).ravel(),256,[0,256]) 
f2_ax3 = fig.add_subplot(spec2[1, 0])
f2_ax3.imshow(cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB))
f2_ax4 = fig.add_subplot(spec2[1, 1])
f2_ax4.hist(cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY).ravel(),256,[0,256]) 

# %%

####################################################################
##################### Convert image to b/w #########################
####################################################################
import cv2
import numpy as np
from PIL import Image

# img = Image.open("images/pbear1.jpg") # open colour image
# img = img.convert('1') # convert image to black and white
# img.show()

originalImage = cv2.imread('images/bird1.jpg')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
# blackAndWhiteImage = cv2.adaptiveThreshold(grayImage, 127, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 5)


im_arr = np.asarray(grayImage)
img = Image.fromarray(im_arr)
# img.save("images/grayscale_image.jpg")
img.show()

# %%
####################################################################
############# Convert image to b/w and add noise ###################
####################################################################
import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage.util import random_noise

# img = Image.open("images/pbear1.jpg") # open colour image
# img = img.convert('1') # convert image to black and white
# img.show()

originalImage = cv2.imread('images/bird1.jpg')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
# blackAndWhiteImage = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 5)

im_arr = np.asarray(grayImage)

# random_noise() method will convert image in [0, 255] to [0, 1.0],
# inherently it use np.random.normal() to create normal distribution
# and adds the generated noised back to image
noise_img = random_noise(im_arr, mode='gaussian', var=0.2**2)
noise_img = (255*noise_img).astype(np.uint8)

# row, col = grayImage.shape
# gauss = np.random.normal(10,10,(row,col))
# noisy = grayImage + gauss


fig1 = plt.figure(figsize=(9,3),constrained_layout=True)
spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)
fig1_sub1 = fig1.add_subplot(spec1[0, 0])
fig1_sub1.imshow(grayImage, cmap='gray')
fig1_sub1.set_xlabel('image width')
fig1_sub1.set_ylabel('image height')

fig1_sub2 = fig1.add_subplot(spec1[0, 1])
fig1_sub2.hist(grayImage.ravel(),256,[0,256])
fig1_sub2.set_xlabel('pixel value')
fig1_sub2.set_ylabel('count')

fig1.savefig('plots/bird_grayscale.png', bbox_inches='tight')

fig2 = plt.figure(figsize=(9,3),constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig2)
fig2_sub1 = fig2.add_subplot(spec2[0, 0])
fig2_sub1.imshow(noise_img, cmap='gray')
fig2_sub1.set_xlabel('Image width')
fig2_sub1.set_ylabel('image height')

fig2_sub2 = fig2.add_subplot(spec2[0, 1])
fig2_sub2.hist(noise_img.ravel(),256,[0,256]) 
fig2_sub2.set_xlabel('pixel value')
fig2_sub2.set_ylabel('count')

fig2.savefig('plots/bird_grayscale_noise.png', bbox_inches='tight')

fig3 = plt.figure(figsize=(9,3),constrained_layout=True)
spec3 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig3)
fig3_sub1 = fig3.add_subplot(spec3[0, 0])
fig3_sub1.imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB), cmap='gray')
fig3_sub1.set_xlabel('image width')
fig3_sub1.set_ylabel('image height')

fig3_sub2 = fig3.add_subplot(spec3[0, 1])
fig3_sub2.hist(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB).ravel(),256,[0,256])
fig3_sub2.set_xlabel('pixel value')
fig3_sub2.set_ylabel('count')

fig3.savefig('plots/bird_original.png', bbox_inches='tight')

# %%

### Another way of converting to grayscale using the Recommendation BT.601-7
import cv2
from skimage.util import random_noise
import numpy as np
from PIL import Image
from skimage.filters import threshold_yen
from skimage.io import imread, imsave, imshow_collection
from skimage.exposure import rescale_intensity, equalize_adapthist
import matplotlib.pyplot as plt
import skimage as ski
# Imported PIL Library from PIL import Image

# Open an Image
def open_image(path):
  newImage = Image.open(path)
  return newImage

# Save Image
def save_image(image, path):
  image.save(path, 'png')


# Create a new image with the given size
def create_image(i, j):
  image = Image.new("RGB", (i, j), "white")
  return image


# Get the pixel from the given image
def get_pixel(image, i, j):
  # Inside image bounds?
  width, height = image.size
  if i > width or j > height:
    return None

  # Get Pixel
  pixel = image.getpixel((i, j))
  return pixel

# Create a Grayscale version of the image
def convert_grayscale(image):
  # Get size
  width, height = image.size

  # Create new Image and a Pixel Map
  new = create_image(width, height)
  pixels = new.load()

  # Transform to grayscale
  for i in range(width):
    for j in range(height):
      # Get Pixel
      pixel = get_pixel(image, i, j)

      # Get R, G, B values (This are int from 0 to 255)
      red =   pixel[0]
      green = pixel[1]
      blue =  pixel[2]

      # Transform to grayscale
      gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

      # Set Pixel in new image
      pixels[i, j] = (int(gray), int(gray), int(gray))

  # Return new image
  return new



# Load Image (JPEG/JPG needs libjpeg to load)
original = open_image('images/pbear1.jpg')

# Example Pixel Color
print('Color: ' + str(get_pixel(original, 0, 0)))

# Convert to Grayscale and save
new = convert_grayscale(original)

new.show()



# %%
