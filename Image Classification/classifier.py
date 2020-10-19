"""
Computer Vision - Purdue University - Homework 7

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Oct 19, 2020


[TO RUN CODE]: python3 classifier.py
Output:
    [labels]: Predictions for the input images
"""

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import copy
import math
import glob
import BitVector
import pickle
import os
from collections import Counter

class Imageclassifier:

    def __init__(self, training_directory, testing_directory, parameterR, parameterP, kvalue):
        self.classdict = dict()
        self.imagedict = dict()
        self.binarypatterndict = dict()
        self.classcount = len(os.listdir(training_directory))
        for element in os.listdir(training_directory):
            self.classdict[element] = len(os.listdir(training_directory+'/'+element))
            templist = []
            for image in sorted(os.listdir(training_directory+'/'+element)):
                templist.append(cv.resize(cv.imread(training_directory+'/'+element+'/'+image, cv.IMREAD_GRAYSCALE), (640, 480)))
            self.imagedict[element]=templist
        self.parameterR = parameterR
        self.parameterP = parameterP
        self.kneighbors = kvalue
        print(self.imagedict['beach'][0].shape)

    def get_pixel_value(self, queuetuple, delu, delv, centerX, centerY):
        """
        This function implements the bilinear interpolation method used to get the pixel value
        or the grey value at the point p
        :param queuetuple: Location of the image
        :param delu: change in value in x direction
        :param delv: change in value in y direction
        :param centerX: the point at the center of the circle under consideration
        :param centerY: the point at the center of the circle under consideration
        :return: greylevel at the point pß
        """
        image = self.imagedict[queuetuple[0]][queuetuple[1]]
        if (delu < 0.001) and (delv < 0.001):
            interpolated_greylevel = float(image[centerX][centerY])
        elif (delv < 0.001):
            interpolated_greylevel = (1 - delu) * image[centerX][centerY] + delu * image[centerX + 1][centerY]
        elif (delu < 0.001):
            interpolated_greylevel = (1 - delv) * image[centerX][centerY] + delv * image[centerX][centerY + 1]
        else:
            interpolated_greylevel = (1 - delu) * (1 - delv) * image[centerX][centerY] + (1 - delu) * delv * image[centerX][centerY + 1] + delu * delv * \
                          image[centerX + 1][centerY + 1] + delu * (1 - delv) * image[centerX + 1][centerY]
        return interpolated_greylevel

    def generate_texture_feature(self, queuetuple):

        LBP_hist = {bins: 0 for bins in range(self.parameterP + 2)}
        print(LBP_hist)
        greyimage = self.imagedict[queuetuple[0]][queuetuple[1]]
        for row in range(self.parameterR, greyimage.shape[0]-self.parameterR):
            for column in range(self.parameterR, greyimage.shape[1]- self.parameterR):
                binarypatternforpoint = []
                for pointnumber in range(self.parameterP):
                    delu = self.parameterR * math.cos(2 * math.pi * pointnumber / self.parameterP)
                    delu = self.parameterR * math.sin(2 * math.pi * pointnumber / self.parameterP)
                    if abs(del_u) < 0.001: del_u = 0.0
                    if abs(del_v) < 0.001: del_v = 0.0
                    greylevel = self.get_pixel_value(queuetuple, delu, del_v, int(row+delu), int(column+del_v))
                    if greylevel >= greyimage[row][column]:
                        binarypatternforpoint.append(1)
                    else:
                        binarypatternforpoint.append(0)

                bv = BitVector.BitVector(bitlist=binarypatternforpoint)
                intvals_for_circular_shifts = [int(bv << 1) for _ in range(P)]
                minbv = BitVector.BitVector(intVal=min(intvals_for_circular_shifts), size=P)
                bvruns = minbv.runs()
                encoding = None



if __name__ == "__main__":
    tester = Imageclassifier("imagesDatabaseHW7/training", "ImageDatabaseHW7/testing", 1, 8, 5)
    tester.generate_texture_feature(('beach', 0))
