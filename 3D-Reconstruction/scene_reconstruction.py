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
        self.count = 0
        self.parameter_dict = dict()
        for image_path_index in tqdm(range(len(image_paths)), desc='Image Load'):
            file = cv.imread(image_paths[image_path_index])
            self.image_pair.append(file)
            self.image_specs.append(file.shape)

    def schedule(self):
        self.getROIFromUser()
        self.process_points()
        self.process_rectification()

    def process_rectification(self):
        x1,x2 = list(map(lambda  x: x[0], self.left_manual_points)),list(map(lambda  x: x[0], self.right_manual_points))
        y1,y2 = list(map(lambda  x: x[1], self.left_manual_points)),list(map(lambda  x: x[1], self.right_manual_points))
        mux_1,mux_2 = np.mean(x1),np.mean(x2)
        muy_1,muy_2 = np.mean(y1), np.mean(y2)
        tx1,tx2 = np.square(x1 - mux_1),np.square(x2 - mux_2)
        ty1,ty2 = np.square(y1 - muy_1),np.square(y2 - muy_2)
        m1,m2 = (1.0 / len(self.left_manual_points)) * np.sum(np.sqrt(np.add(tx1,tx2))),(1.0 / len(self.right_manual_points)) * np.sum(np.sqrt(np.add(ty1,ty2)))
        s1,s2 = np.sqrt(2)/m1,np.sqrt(2)/m2
        x1,x2 = -1 * s1 * mux_1,-1 * s2 * mux_2
        y1,y2 = -1 * s1 * muy_1,-1 * s2 * muy_2
        T1,T2 = np.array([[s1, 0, x1], [0, s1, y1], [0, 0, 1]]),np.array([[s2, 0, x2], [0, s2, y2], [0, 0, 1]])
        self.parameter_dict['T1'] = T1
        self.parameter_dict['T2'] = T2

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
            self.roiCoordinates.append((int(x), int(y)))
            if self.count % 2 == 0:
                cv.circle(self.image,(int(x), int(y)),5,[0,255,255],3)
            else:
                cv.circle(self.image, (int(x), int(y)), 5, [0, 255, 255], 3)
                cv.line(self.image,self.roiCoordinates[self.count-1],(int(x), int(y)),[0,255,0],3)
            self.count +=1

    def getROIFromUser(self):
        """
        [This function is responsible for taking the regions of interests from the user for all the 4 pictures in order]

        """
        self.roiList = []
        cv.namedWindow('Select ROI')
        cv.setMouseCallback('Select ROI', self.append_points)
        self.image = np.hstack((self.image_pair[0],self.image_pair[1]))
        while (True):
            cv.imshow('Select ROI', self.image)
            k = cv.waitKey(1) & 0xFF
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.imwrite('result_1.jpg',self.image)


if __name__ == "__main__":
    tester = Reconstruct(['Task2_Images/Left.jpg','Task2_Images/Right.jpg'])
    tester.schedule()