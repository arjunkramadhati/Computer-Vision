"""
Computer Vision - Purdue University - Homework 4

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: September 27, 2020


[TO RUN CODE]: python3 cornerdetector.py
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

    def __init__(self, image_addresses, scale, kvalue=0):
        self.image_addresses = image_addresses
        self.scale = scale
        self.originalImages = []
        self.grayscaleImages = []
        self.imagesizes = []
        self.filters = {}
        self.cornerpointdict = {}
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

    def filter_corner_points(self, rscore, windowsize, queueImage, tag):
        window = int(windowsize / 2)
        for column in range(window, self.grayscaleImages[queueImage].shape[1] - window, 1):
            for row in range(window, self.grayscaleImages[queueImage].shape[0] - window, 1):
                panwindow = rscore[row - window:row + window +1, column - window : column + window +1]
                print(panwindow.shape)
                if rscore[row, column] == np.amax(panwindow):
                    pass
                else:
                    rscore[row, column] = 0

        self.cornerpointdict[tag] = np.asarray(np.where(rscore > 0))
        print(len(np.asarray(np.where(rscore > 0))[0]))

    def draw_corner_points(self, queueImage, tag):

        points = self.cornerpointdict[tag].flatten()
        image = self.originalImages[queueImage]
        for index in range(len(points)):
            if index == len(points) - 2:
                break
            cv.circle(image, (points[index+2], points[index]), 4, [255, 255, 255], 10)
        return image


    def determine_corners(self, type, queueImage, tag):
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
            detvalue = (sumofdxsquared * sumofdysquared) - (sumofdxdy * sumofdxdy)
            tracevalue = sumofdysquared * sumofdxsquared
            if self.kvalue == 0:
                self.kvalue = detvalue / (tracevalue * tracevalue + 0.000001)
                self.kvalue = np.sum(self.kvalue) / (
                            self.grayscaleImages[queueImage].shape[0] * self.grayscaleImages[queueImage].shape[1])
            Rscore = detvalue - (self.kvalue * tracevalue * tracevalue)
            Rscore = np.where(Rscore < 0, 0, Rscore)
            print(Rscore.shape)
            self.filter_corner_points(Rscore, 29, queueImage, tag)

if __name__ == "__main__":
    tester = FeatureOperator(['hw4_Task1_Images/pair1/1.jpg'], 1.407)
    tester.build_haar_filter()
    tester.determine_corners(1, 0, "Harris")
    image = tester.draw_corner_points(0,"Harris")
    cv.imwrite("2.jpg", image)