"""
Computer Vision - Purdue University - Homework 6

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Oct 5, 2020


[TO RUN CODE]: python3 segmentimages.py
Output:
    [jpg]: Segmented image which shows the foreground separated from the background.
"""

import cv2 as cv
from matplotlib import pyplot as plt
import math
import numpy as np
import random
import time
from scipy import signal as sg
import tqdm
import copy
import threading
from scipy.optimize import least_squares
from scipy.optimize import minimize

class ImageSegmentation:
    def __init__(self, image_addresses):
        self.image_addresses = image_addresses
        self.originalImages = []
        self.grayscaleImages = []
        self.rgbchannelsdict = {}
        for i in range(len(self.image_addresses)):
            self.originalImages.append(cv.resize(cv.imread(self.image_addresses[i], cv.IMREAD_COLOR), (640, 480)))
            self.grayscaleImages.append(
                cv.resize(cv.cvtColor(cv.imread(self.image_addresses[i]), cv.COLOR_BGR2GRAY), (640, 480)))

    def split_channels(self, inputstyle = 'BGR', gaussianblur = True):
        for queue in range(len(self.originalImages)):
            if inputstyle == 'BGR':
                r,g,b= self.originalImages[queue][:,:,2], self.originalImages[queue][:,:,1], self.originalImages[queue][:,:,0]
            elif inputstyle =='RGB':
                r, g, b = self.originalImages[queue][:, :, 0], self.originalImages[queue][:, :, 1], self.originalImages[queue][:, :, 2]
            if gaussianblur:
                r,g,b = cv.GaussianBlur(r, (5,5), 0), cv.GaussianBlur(g, (5,5), 0), cv.GaussianBlur(b, (5,5), 0)
            self.rgbchannelsdict[queue]={'R': r, 'G': g, 'B': b}

    def run_otsu(self, imagequeue):
        for channel in ['R','G','B']:
            image = self.rgbchannelsdict[imagequeue][channel]
            channelhistogram = cv.calcHist([np.uint8(image)],[0], None, [256], [0, 256])
            levels = np.reshape(np.add(range(256) , 1) , (256, 1))
            maxlambda = -1
            otsucutoff = -1
            plt.hist(image.ravel(), 256, [0, 256]);
            plt.show()

            for i in range(len(channelhistogram)):
                m0k = np.sum(channelhistogram[:i])/np.sum(channelhistogram)
                m1k = np.sum(np.multiply(channelhistogram[:i],levels[:i]))/np.sum(channelhistogram)
                m11k = np.sum(np.multiply(channelhistogram[i:],levels[i:]))/np.sum(channelhistogram)
                omega0 = m0k
                omega1 = 1 - m0k
                mu0 = m1k/omega0
                mu1 = m11k/omega1
                sqauredifference = np.square(mu1-mu0)
                lambdavalue = omega0*omega1*sqauredifference
                if lambdavalue > maxlambda:
                    maxlambda = lambdavalue
                    otsucutoff = i
            print(otsucutoff)





if __name__ =='__main__':
    tester = ImageSegmentation(['hw6_images/cat.jpg','hw6_images/pigeon.jpg','hw6_images/Red-Fox_.jpg'])
    tester.split_channels()
    tester.run_otsu(2)
