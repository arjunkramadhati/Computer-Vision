"""
Computer Vision - Purdue University - Homework 10

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Oct 19, 2020


[TO RUN CODE]: python3 scene_reconstruction.py
"""

import re
import glob
import pickle
import cv2 as cv
import numpy as np
from tqdm import tqdm
from sklearn import svm
from scipy import signal
from sklearn.model_selection import train_test_split
from PIL import Image,ImageFont,ImageDraw


class Reconstruct:

    def __init__(self, image_paths):
        self.image_pair = list()
        self.roiList = list()
        self.roiCoordinates = list()
        self.image_specs = list()
        self.left_manual_points = list()
        self.right_manual_points = list()
        for image_path_index in tqdm(range(len(image_paths)), desc='Image Load'):
            file = cv.imread(image_paths[image_path_index])
            self.image_pair.append(file)
            self.image_specs.append(file.shape)

    def schedule(self):
        self.getROIFromUser()
        self.process_points()

    def process_points(self):
        for element_index in tqdm(range(len(self.roiCoordinates)),desc='Point process'):
            if self.roiCoordinates[element_index][0]>self.image_specs[0][1]:
                point = (self.roiCoordinates[element_index][0]-self.image_specs[0][1],self.roiCoordinates[element_index][1])
                self.right_manual_points.append(point)
            else:
                self.left_manual_points.append(self.roiCoordinates[element_index])

    def append_points(self, event, x, y, flags, param):
        """
        [This function is called every time the mouse left button is clicked - It records the (x,y) coordinates of the click location]

        """
        if event == cv.EVENT_LBUTTONDOWN:
            self.roiCoordinates.append((float(x), float(y)))

    def getROIFromUser(self):
        """
        [This function is responsible for taking the regions of interests from the user for all the 4 pictures in order]

        """
        self.roiList = []
        cv.namedWindow('Select ROI')

        cv.setMouseCallback('Select ROI', self.append_points)
        image = np.hstack((self.image_pair[0],self.image_pair[1]))
        while (True):
            cv.imshow('Select ROI', image)
            k = cv.waitKey(1) & 0xFF
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    tester = Reconstruct(['Task2_Images/Left.jpg','Task2_Images/Right.jpg'])
    tester.schedule()