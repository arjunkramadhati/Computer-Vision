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
from scipy.linalg import null_space
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
        self.reference_center = np.array([[self.image_specs[0][1] / 2.0, self.image_specs[0][0] / 2.0, 1]])
        print("----------------------------")
        print("Initialization complete")
        print("----------------------------")

    def schedule(self):
        #1Get interest points from user
        print('hello')
        self.getROIFromUser()
        #2Process the points: Segregate them based on the left and right images
        self.process_points()
        #3Perform stereo rectification
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
        self.parameter_dict['NLeft'] = np.matmul(T1, np.array(list(map(lambda x: [x[0], x[1], 1], self.left_manual_points))).T)
        self.parameter_dict['NRight'] = np.matmul(T2, np.array(
            list(map(lambda x: [x[0], x[1], 1], self.right_manual_points))).T)
        NLeftT= self.parameter_dict['NLeft'].T
        NRightT = self.parameter_dict['NRight'].T
        matrixA = self.getA(NLeftT,NRightT)
        u,d,vt = np.linalg.svd(matrixA)
        v=vt.T
        initial_F_estimate = v[:,v.shape[1]-1]
        assert len(initial_F_estimate)==9
        initial_F_estimate=initial_F_estimate.reshape(3,3)
        initial_F_estimate = self.reinforce_F_estimate(initial_F_estimate,T1,T2)
        self.parameter_dict['F_beta']=initial_F_estimate
        print("----------------------------")
        print("Initial F estimate complete")
        print("----------------------------")
        e_one,e_two = self.get_nulls(self.parameter_dict['F_beta'])
        H2 = self.get_homography(e_one=e_one,e_two=e_two,type='H2')
        center_value = self.get_updated_center(homography=H2)
        second_T_value = np.array([[1, 0, (self.image_specs[0][1] / 2.0) - center_value[0]], [0, 1, (self.image_specs[0][0] / 2.0) - center_value[1]], [0, 0, 1]])
        H2 = np.matmul(second_T_value,H2)
        H1 = self.get_homography(e_one=e_one, e_two=e_two, type='H1')
        center_value_2 = self.get_updated_center(homography=H1)
        first_T_value = np.array([[1, 0, (self.image_specs[0][1] / 2.0) - center_value_2[0]], [0, 1, (self.image_specs[0][0] / 2.0) - center_value_2[1]], [0, 0, 1]])
        H1 = np.matmul(first_T_value,H1)
        self.parameter_dict['H1&H2']=[H1,H2]
        P_dash_value = self.get_P_values(e_two=e_two,F=self.parameter_dict['F_beta'])
        print("----------------------------")
        print("H1 & H2 estimation complete")
        print("----------------------------")
        self.rectify_image()
        print("----------------------------")
        print("Individual rectification complete")
        print("----------------------------")

    def rectify_image(self):

        for index, element in self.parameter_dict['H1&H2']:
            correlation = self.get_correlation(index,element)
            values = [min(correlation[0]), min(correlation[1]),max(correlation[0]), max(correlation[1])]
            d_im = np.array(values[1]) - np.array(values[0])
            d_im = [int(d_im[0]), int(d_im[1])]
            scale = np.array([[self.image_specs[index][1] / d_im[0], 0, 0], [0, self.image_specs[index][0] / d_im[1], 0], [0, 0, 1]])
            H = np.matmul(scale, element)
            correlation = self.get_correlation(index,H)
            values_2 = [min(correlation[0]), min(correlation[1])]
            d_im = values_2
            d_im = [int(d_im[0]), int(d_im[1])]
            T_value = np.array([[1, 0, -1 * d_im[0] + 1], [0, 1, -1 * d_im[1] + 1], [0, 0, 1]])
            homography_n = np.matmul(T_value, H)
            inverse_homography = np.linalg.pinv(homography_n)
            result_image = self.create_image(index=index,H=inverse_homography)
            self.parameter_dict['Rectified_Params'+str(index)] = [result_image, homography_n]


    def create_image(self, index, H):
        result_image = np.zeros((self.image_specs[index][0], self.image_specs[index][1], 3))
        for row in range(self.image_specs[index][0]):
            for column in range(self.image_specs[index][1]):
                temp_variable = np.matmul(H, np.array([[row],[column],[1]]))
                temp_variable = temp_variable/temp_variable[2]
                if temp_variable[0] >= 0 and temp_variable[0] < self.image_specs[index][0] and int(temp_variable[1]) >= 0 and int(temp_variable[1]) < self.image_specs[index][1]:
                    result_image[row, column] = self.image_pair[index][int(temp_variable[0]), int(temp_variable[1])]
        cv.imwrite('rectified_'+str(index)+'.jpg',result_image)
        return result_image

    def get_correlation(self, index, element):
        correlation = np.matmul(element, np.array(
            [[0, self.image_specs[index][1], 0, self.image_specs[index][1]],
             [0, 0, self.image_specs[index][0], self.image_specs[index][0]], [1, 1, 1, 1]]))
        return (correlation / correlation[2])


    def get_P_values(self, e_two, F ):
        return np.append(np.matmul(
            np.array(
                [[0, -1 * e_two[2], e_two[1]], [e_two[2], 0, -1 * e_two[0]], [-1 * e_two[1], e_two[0], 0]]),
            F
        ),
        np.array([e_two[0], e_two[1], e_two[2]])
        ,
        axis=1)


    def get_updated_center(self, homography):
        center = np.matmul(homography,self.reference_center.T)
        return center/center[2]

    def get_homography(self,e_one, e_two, type):
        if type == 'H2':
            theta_value = self.get_theta(e_one=e_one,e_two=e_two,image_index=0, type='2')
            F_value = (np.cos(theta_value)*(e_two[0]-self.image_specs[0][1]/2.0)-np.sin(theta_value)*(e_two[1]-self.image_specs[0][0]/2.0))[0]
            R_value = np.array([[np.cos(theta_value)[0], -1 * np.sin(theta_value)[0], 0], [np.sin(theta_value)[0], np.cos(theta_value)[0], 0], [0, 0, 1]])
            T_value = np.array([[1, 0, -1 * self.image_specs[0][1] / 2.0], [0, 1, -1 * self.image_specs[0][0] / 2.0], [0, 0, 1]])
            G_value = np.array([[1, 0, 0], [0, 1, 0], [-1.0 / F_value, 0, 1]])
            H = np.matmul(np.matmul(G_value,R_value),T_value)
            assert H.shape == (3,3)
            return H
        elif type == ' H1':
            theta_value = self.get_theta(e_one=e_one,e_two=e_two,image_index=0, type='1')
            F_value = (np.cos(theta_value) * (e_one[0] - self.image_specs[0][1] / 2.0) - np.sin(theta_value) * (e_one[1] - self.image_specs[0][0] / 2.0))[0]
            R_value = np.array([[np.cos(theta_value)[0], -1 * np.sin(theta_value)[0], 0], [np.sin(theta_value)[0], np.cos(theta_value)[0], 0], [0, 0, 1]])
            T_value = np.array([[1, 0, -1 * self.image_specs[0][1] / 2.0], [0, 1, -1 * self.image_specs[0][0] / 2.0], [0, 0, 1]])
            G_value = np.array([[1, 0, 0], [0, 1, 0], [-1.0 / F_value, 0, 1]])
            H = np.matmul(np.matmul(G_value,R_value),T_value)
            assert H.shape == (3,3)
            return H


    def get_theta(self, e_one, e_two, image_index, type):
        if type == '2':
            return np.arctan(-1*(e_two[1]-self.image_specs[image_index][0]/2.0)/(e_two[0]-self.image_specs[image_index][1]/2.0))
        elif type == '1':
            return np.arctan(-1 * (e_one[1] - self.image_specs[image_index][0] / 2.0) / (
                        e_one[0] - self.image_specs[image_index][1] / 2.0))

    def get_nulls(self, F):
        e_one = null_space(F)
        e_two = null_space(F.T)
        e_one = e_one/e_one[2]
        e_two = e_two/e_two[2]
        return e_one, e_two

    def reinforce_F_estimate(self, F, T1,T2):
        u,d,vt = np.linalg.svd(F)
        d=np.array([[d[0],0,0],[0,d[1],0],[0,0,0]])
        return np.matmul(np.matmul(T2.T, np.matmul(np.matmul(u, d), vt)), T1)

    def getA(self,point_left, point_right):
        matrixA = list()
        for index in range(len(point_left)):
            value = [self.right_manual_points[index][0]*self.left_manual_points[index][0],
                     self.right_manual_points[index][0]*self.left_manual_points[index][1],
                     self.right_manual_points[index][0],self.right_manual_points[index][1]*self.left_manual_points[index][0],
                     self.right_manual_points[index][1]*self.left_manual_points[index][1],
                     self.right_manual_points[index][1],
                     self.left_manual_points[index][0],
                     self.right_manual_points[index][1],
                     1.0]
            matrixA.append(value)
        return matrixA

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
        print('sd')
        cv.namedWindow('Select ROI')
        print('named')
        cv.setMouseCallback('Select ROI', self.append_points)
        self.image = np.hstack((self.image_pair[0],self.image_pair[1]))
        print('e')
        while(True):
            cv.imshow('Select ROI', self.image)
            k = cv.waitKey(1) & 0xFF
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.imwrite('result_1.jpg',self.image)


if __name__ == "__main__":
    tester = Reconstruct(['Task2_Images/Left.jpg','Task2_Images/Right.jpg'])
    tester.schedule()