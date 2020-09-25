"""
Computer Vision - Purdue University - Homework 3

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: September 21, 2020


[TO RUN CODE]: python3 removeDistortion.py
The code displays the pictures. The user will have to select the ROI points manually in the PQRS fashion.
P ------- Q
|         |
|         |
|         |
R ------- S

Output:
    [jpg]: [Transformed images]
"""

import cv2 as cv
import math
import numpy as np
import time
from scipy import signal as sg
import tqdm


class FeatureOperator:

    def __init__(self, image_addresses, scale, kvalue =0):
        self.image_addresses = image_addresses
        self.scale = scale
        self.originalImages = []
        self.grayscaleImages = []
        self.imagesizes = []
        self.filters = {}
        self.kvalue = kvalue
        for i in range(len(self.image_addresses)):
            self.originalImages.append(cv.imread(self.image_addresses[i]))
            self.grayscaleImages.append(cv.cvtColor(cv.imread(self.image_addresses[i]), cv.COLOR_BGR2GRAY))

    def build_haar_filter(self):
        mvalue = int(np.ceil(4 * self.scale))
        mvalue = mvalue + 1 if (mvalue % 2) > 0 else mvalue
        blankfilter = np.ones((mvalue, mvalue))
        blankfilter[:, :int(mvalue / 2)] = -1
        self.filters["HaarFilterX"] = blankfilter
        blankfilter = np.ones((mvalue, mvalue))
        blankfilter[int(mvalue / 2):, :] = - 1
        self.filters["HaarFilterY"] = blankfilter

    def determine_corners(self, type, queueImage):

        if type == 1:
            # Harris Corner Method
            dx = sg.convolve2d(self.grayscaleImages[queueImage], self.filters["HaarFilterX"], mode='same')
            dy = sg.convolve2d(self.grayscaleImages[queueImage], self.filters["HaarFilterY"], mode='same')
            dxsquared = dx * dx
            dysquared = dy * dy
            dxy = dx * dy
            windowsize = int(5 * self.scale)
            windowsize = windowsize if (windowsize % 2) > 0 else windowsize + 1
            window = np.ones((windowsize, windowsize))
            sumofdxsquared = sg.convolve2d(dxsquared, window, mode='same')
            sumofdysquared = sg.convolve2d(dysquared, window, mode='same')
            sumofdxdy = sg.convolve2d(dxy, window, mode='same')
            detvalue = (sumofdxsquared*sumofdysquared) - (sumofdxdy*sumofdxdy)
            tracevalue = sumofdysquared * sumofdxsquared
            if self.kvalue == 0:
                self.kvalue = detvalue/(tracevalue*tracevalue)
                self.kvalue = np.sum(self.kvalue)/(self.grayscaleImages[queueImage].shape[0]*self.grayscaleImages[queueImage].shape[1])

            Rscore = detvalue - (self.kvalue*tracevalue*tracevalue)


