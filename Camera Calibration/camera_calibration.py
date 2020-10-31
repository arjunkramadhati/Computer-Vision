"""
Computer Vision - Purdue University - Homework 9

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Nov 2, 2020


[TO RUN CODE]: python3 camera_calibration.py
"""
import re
import glob
import pickle
from tqdm import tqdm
import cv2 as cv
import numpy as np
from sklearn import svm
from scipy import signal
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pylab import *
from scipy . optimize import leastsq


class Calibrate:
    def __init__(self, image_path):

        print("Initializing Calibration process...")
        self.image_path = glob.glob(image_path)
        print("Loading image from path " + image_path)
        self.color_images_dict = dict()
        self.gray_images_dict = dict()
        self.lines_dict = dict()
        self.corner_size = (8,10)
        for index, element in enumerate(tqdm(self.image_path)):
            image = cv.imread(element)
            self.color_images_dict[index] = image
            self.gray_images_dict[index] = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        print("Image load complete...")
        print("Initialization complete")
        print("-------------------------------------")
        print("-------------------------------------")

    def get_line(self, rho, theta):
        proportion_one = np.cos(theta)
        proportion_two = np.sin(theta)
        centerX = rho*proportion_one
        centerY = rho*proportion_two
        p1 = int(centerX + 1000*(-proportion_two))
        p2 = int(centerY + 1000*(proportion_one))
        p3 = int(centerX - 1000*(-proportion_two))
        p4 = int(centerY - 1000*(proportion_one))
        return (p1,p2),(p3,p4)

    def extract_lines(self, cutoff = 50, output_path='Files/calibration_output/edges_lines/'):
        print("Detecting Edges and Lines...")
        print("-------------------------------------")
        for key in tqdm(range(len(self.color_images_dict.keys()))):
            color = (self.color_images_dict[key].copy())/2
            edges = cv.Canny(cv.GaussianBlur(self.gray_images_dict[key],(5,5),0),2500, 4000, apertureSize=5)
            color[edges!=0] = (0,0,255)
            hline = cv.HoughLines(edges,1, np.pi/180, cutoff)
            for line in hline:
                for rho, theta in line:
                    point_one, point_two = self.get_line(rho, theta)
                    cv.line(color, point_one, point_two, (0, 255, 0), 3)
            self.lines_dict[key] = hline
            cv.imwrite(output_path+str(key)+'.jpg', color)
        print("Line extraction complete...")
        print("-------------------------------------")

    def filter_list(self, hlist, vlist, distance_cutoff = 15):
        sorted_hlist = sorted(hlist, key=lambda hline:hline[0][0]*np.sin(hline[0][1]),reverse=False)
        sorted_vlist = sorted(vlist, key= lambda vline:vline[0][0]*np.cos(vline[0][1]), reverse=False)
        filtered_hlist = []
        filtered_vlist = []
        for index in range(len(sorted_hlist)):
            rho = sorted_hlist[index][0]
            theta = sorted_hlist[index][1]
            if index == 0:
                filtered_hlist.append(sorted_hlist[index])
            elif abs(rho * np.sin(theta) - filtered_hlist[-1][0] * np.sin(filtered_hlist[-1][1])) > distance_cutoff:
                filtered_hlist.append(sorted_hlist[index])
        for index in range(len(sorted_vlist)):
            rho = sorted_vlist[index][0]
            theta = sorted_vlist[index][1]
            if index == 0:
                filtered_vlist.append(sorted_vlist[index])
            elif abs(rho * np.sin(theta) - filtered_vlist[-1:][0] * np.sin(filtered_vlist[-1:][1])) > distance_cutoff:
                filtered_vlist.append(sorted_vlist[index])

        return filtered_hlist,filtered_vlist

    def extract_corners(self):
        print('Extracting corners...')
        print("-------------------------------------")
        for key in tqdm(range(len(self.color_images_dict.keys()))):
            horizontal_line_list = []
            vertical_line_list = []
            color = self.color_images_dict[key].copy()
            lines = self.lines_dict[key]
            for line in lines:
                for rho, theta in line:
                    theta -= np.pi/2
                    if abs(theta)<(np.pi/4):
                        horizontal_line_list.append(line)
                    elif abs(theta)>=(np.pi/4):
                        vertical_line_list.append(line)
            assert(len(horizontal_line_list)+len(vertical_line_list) == len(lines))
            horizontal_line_list, vertical_line_list = self.filter_list(horizontal_line_list, vertical_line_list)










if __name__ == "__main__":
    tester = Calibrate('./Files/Dataset1/*')
    tester.extract_lines()
    tester.extract_corners()




