# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:19:58 2020

@author: Vibhav
"""
import cv2, os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img_url = 'C:\\Users\\Vibhav\\Desktop\\Embedded_Systems\\Year1_Q4\\CV_DL\\project\\UdaCity_IKV\\How_to_simulate_a_self_driving_car-master\\data\\IMG\\center_2020_06_03_22_17_47_732.jpg'

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 65, 320, 3

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[0:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    #image = resize(image)
    #image = rgb2yuv(image)
    return image

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    #x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    #x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    #xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    height, width = image.shape[:2]
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = int(width * np.random.rand()), 0
    x2, y2 = int(width * np.random.rand()), height
    xm, ym = np.mgrid[0:height, 0:width]
    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

img = mpimg.imread(img_url)
plt.figure(figsize=(16, 16))
plt.suptitle('Before and After Processing')
plt.subplot(1, 2, 1)
#plt.imshow(train_images[5000])
plt.imshow(img)
plt.title(' Before')
#Data Pre-Processing
img = preprocess(img)
#img = random_brightness(img)

plt.subplot(1, 2, 2)
#plt.imshow(train_images[5000])
plt.imshow(img)
plt.title('After')
plt.show()
