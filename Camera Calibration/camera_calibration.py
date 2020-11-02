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
        self.corner_list = []
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

    def filter_lines(self, houghlines):
        final_hough_list = np.asarray(houghlines).copy()
        final_hlist_hesse = []
        final_vlist_hesse = []
        for index in range(len(final_hough_list)):
            individual_final_list = np.array(final_hough_list[index]).tolist()
            individual_final_list_h = individual_final_list[0:10]
            individual_final_list_h.sort(key=lambda item:item[0][0])
            final_hlist_hesse.append(individual_final_list_h)
            individual_final_list_v = individual_final_list[10:]
            individual_final_list_v.sort(key=lambda item:abs(item[0][0]))
            final_vlist_hesse.append(individual_final_list_v)
        return final_hlist_hesse, final_vlist_hesse

    def filter_list(self, hlist, vlist, distance_cutoffH = 100,distance_cutoffV = 100):
        filtered_hlist = []
        filtered_vlist = []
        while(len(filtered_hlist)<10):
            distance_cutoffH -=0.05
            filtered_hlist = []
            for index in range(len(hlist)):
                selectedline = hlist[index][0][0]
                reject = 0
                for line in filtered_hlist:
                    if abs(abs(line[0][0])- abs(selectedline))<distance_cutoffH:
                        reject = 1
                if reject ==0:
                    filtered_hlist.append(hlist[index])

        while(len(filtered_vlist)<8):
            distance_cutoffV -=0.05
            filtered_vlist=[]
            for index in range(len(vlist)):
                selectedline = vlist[index][0][0]
                reject = 0
                for line in filtered_vlist:
                    if abs(abs(line[0][0]) -abs(selectedline)) < distance_cutoffV:
                        reject = 1
                if reject == 0:
                    filtered_vlist.append(vlist[index])

        return filtered_hlist, filtered_vlist

    def extract_corners(self):
        """
        https://stackoverflow.com/a/383527/5087436
        :return:
        """
        print('Extracting corners...')
        print("-------------------------------------")
        linelist = []
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
            assert(len(horizontal_line_list) == 10)
            assert(len(vertical_line_list) == 8)
            linelist.append(horizontal_line_list+vertical_line_list)
        final_horizontal_lines, final_vertical_lines = self.filter_lines(linelist)
        print('[1] Filtered lines')
        print('[2] Corner extraction started')
        corners = []
        for key in tqdm(range(len(self.color_images_dict.keys()))):
            individual_corners = []
            for index_vertical in range(len(final_vertical_lines[key])):
                for index_horizontal in range(len(final_horizontal_lines[key])):
                    rho_horizontal, theta_horizontal = final_horizontal_lines[key][index_horizontal][0]
                    rho_vertical, theta_vertical = final_vertical_lines[key][index_vertical][0]
                    A = np.array([
                        [np.cos(theta_vertical), np.sin(theta_vertical)],
                        [np.cos(theta_horizontal), np.sin(theta_horizontal)]
                    ])
                    B = np.array([[rho_vertical],[rho_horizontal]])
                    cornerX, cornerY = np.linalg.solve(A,B)
                    cornerX, cornerY = int(np.round(cornerX)), int(np.round(cornerY))
                    individual_corners.append([[cornerX,cornerY]])
            corners.append(individual_corners)
        corners_filtered = np.array(np.asarray(corners).copy()).tolist()
        self.corner_list = corners_filtered
        print('Corner extraction complete...')
        print("-------------------------------------")

    def estimate_corner_homography(self):
        print("-------------------------------------")
        print('Estimating Homography for corner refining...')
        print("-------------------------------------")
        H = []
        for key in tqdm(range(len(self.color_images_dict.keys()))):
            matrixA = np.zeros((len(self.corner_list[key])*2, 9))
            for corner_index in range(len(self.corner_list[key])):
                matrixA[2 * key + 0][0] = (key/10)*2.5
                matrixA[2 * key + 0][1] = (key%10)*2.5
                matrixA[2 * key + 0][2] = 1.0
                matrixA[2 * key + 0][3] = 0.0
                matrixA[2 * key + 0][4] = 0.0
                matrixA[2 * key + 0][5] = 0.0
                matrixA[2 * key + 0][6] = -1*((key/10)*2.5)*self.corner_list[key][corner_index][0][0]
                matrixA[2 * key + 0][7] = -1*((key%10)*2.5)*self.corner_list[key][corner_index][0][0]
                matrixA[2 * key + 0][8] = -1*self.corner_list[key][corner_index][0][0]
                matrixA[2 * key + 1][0] = 0.0
                matrixA[2 * key + 1][1] = 0.0
                matrixA[2 * key + 1][2] = 0.0
                matrixA[2 * key + 1][3] = (key/10)*2.5
                matrixA[2 * key + 1][4] = (key%10)*2.5
                matrixA[2 * key + 1][5] = 1.0
                matrixA[2 * key + 1][6] = -1*((key/10)*2.5)*self.corner_list[key][corner_index][0][1]
                matrixA[2 * key + 1][7] = -1 * ((key % 10) * 2.5) * self.corner_list[key][corner_index][0][1]
                matrixA[2 * key + 1][8] = -1*self.corner_list[key][corner_index][0][1]
            homography = np.zeros((3,3))
            umatrix, dmatrix, vmatrixT = np.linalg.svd(matrixA)
            H_matrix = np.transpose(vmatrixT)[:,-1]
            H_matrix = H_matrix/H_matrix[8]
            homography[0][0] = H_matrix[0]
            homography[0][1] = H_matrix[1]
            homography[0][2] = H_matrix[2]
            homography[1][0] = H_matrix[3]
            homography[1][1] = H_matrix[4]
            homography[1][2] = H_matrix[5]
            homography[2][0] = H_matrix[6]
            homography[2][1] = H_matrix[7]
            homography[2][2] = 1.0
            H.append(homography)
        print("-------------------------------------")
        print('Homography estimation complete...')
        print("-------------------------------------")

if __name__ == "__main__":
    tester = Calibrate('./Files/Dataset1/*')
    tester.extract_lines()
    tester.extract_corners()
    tester.estimate_corner_homography()




