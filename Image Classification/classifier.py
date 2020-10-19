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
        self.histogramdict = dict()
        self.classcount = len(os.listdir(training_directory))
        for element in os.listdir(training_directory):
            self.classdict[element] = len(os.listdir(training_directory+'/'+element))
            templist = []
            for image in sorted(os.listdir(training_directory+'/'+element)):
                origimage = cv.imread(training_directory+'/'+element+'/'+image)
                imageread = np.zeros((origimage.shape[0], origimage.shape[1], origimage.shape[2]), dtype='uint8')
                image_gray = np.zeros((origimage.shape[0], origimage.shape[1]), dtype='uint8')
                imageread = cv.imread(training_directory+'/'+element+'/'+image)
                image_gray = cv.cvtColor(imageread, cv.COLOR_BGR2GRAY)
                templist.append(image_gray)
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

    def build_histogram(self, histogram, runs):
        if len(runs) > 2:
            histogram[self.parameterP + 1] += 1
        elif len(runs) == 1 and runs[0][0] == '1':
            histogram[self.parameterP] += 1
        elif len(runs) == 1 and runs[0][0] == '0':
            histogram[0] += 1
        else:
            histogram[len(runs[1])] += 1
        return histogram

    def generate_texture_feature(self, queuetuple):
        """
        This function implements the building of the Local Binary Pattern histogram for the give image
        :param queuetuple: Location of the image in the dictionary
        :return: None. Stores the histogram in a dictionary
        """
        histogram = {bins: 0 for bins in range(self.parameterP + 2)}
        greyimage = self.imagedict[queuetuple[0]][queuetuple[1]]
        for row in range(self.parameterR, greyimage.shape[0]-self.parameterR-1):
            # print(str(row) + " out of " + str(greyimage.shape[0] - self.parameterR - 1))
            for column in range(self.parameterR, greyimage.shape[1]- self.parameterR-1):
                binarypatternforpoint = []
                for pointnumber in range(self.parameterP):
                    delu = self.parameterR * math.cos(2 * math.pi * pointnumber / self.parameterP)
                    delv = self.parameterR * math.sin(2 * math.pi * pointnumber / self.parameterP)
                    if abs(delu) < 0.001: delu = 0.0
                    if abs(delv) < 0.001: delv = 0.0
                    greylevel = self.get_pixel_value(queuetuple, delu, delv, int(row+delu), int(column+delv))
                    if greylevel >= greyimage[row][column]:
                        binarypatternforpoint.append(1)
                    else:
                        binarypatternforpoint.append(0)

                bitvector = BitVector.BitVector(bitlist=binarypatternforpoint)
                intvals_for_circular_shifts = [int(bitvector << 1) for _ in range(self.parameterP)]
                minimum_bit_vector = BitVector.BitVector(intVal=min(intvals_for_circular_shifts), size=self.parameterP)
                runs = minimum_bit_vector.runs()
                histogram = self.build_histogram(histogram, runs)
        self.histogramdict[queuetuple] = histogram
        plt.bar(list(histogram.keys()), histogram.values(), color='b')
        path = 'histograms/' + str(queuetuple[0]) + '/'
        plt.savefig(path + 'Class_{}'.format(queuetuple[0]) + '_ImageNum_{}'.format(int(queuetuple[1])) + '.png')


if __name__ == "__main__":
    tester = Imageclassifier("imagesDatabaseHW7/training", "ImageDatabaseHW7/testing", 1, 8, 5)
    for element in os.listdir("imagesDatabaseHW7/training"):
        for index in range(len(os.listdir("imagesDatabaseHW7/training" + '/' + element))):
            print('training image class: '+element+' __ ' + str(index))
            tester.generate_texture_feature((element, index))
