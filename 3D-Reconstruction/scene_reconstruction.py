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
        self.grey_image_pair = list()
        self.roiList = list()
        self.roiCoordinates = list()
        self.image_specs = list()
        self.left_manual_points = list()
        self.right_manual_points = list()
        self.count = 0
        self.padding = int(31 / 2)
        self.parameter_dict = dict()
        for image_path_index in tqdm(range(len(image_paths)), desc='Image Load'):
            file = cv.imread(image_paths[image_path_index])
            self.image_pair.append(file)
            self.grey_image_pair.append(cv.cvtColor(file, cv.COLOR_BGR2GRAY))
            self.image_specs.append(file.shape)
        self.reference_center = np.array([[self.image_specs[0][1] / 2.0, self.image_specs[0][0] / 2.0, 1]])
        print("----------------------------")
        print("Initialization complete")
        print("----------------------------")

    def schedule(self):
        #1Get interest points from user
        print('hello')
        self.getROIFromUser(type='yes')
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
        F = np.matmul(np.linalg.pinv(self.parameter_dict['Rectified_Params1'][1].T), self.parameter_dict['F_beta']);
        F = np.matmul(F, np.linalg.inv(self.parameter_dict['Rectified_Params0'][1]))
        tp_1 = list(map(lambda x: [x[1], x[0], 1], self.left_manual_points));
        point_one = np.matmul(H1, np.array(tp_1).T)
        point_one /= point_one[2]
        point_one = point_one.T
        point_one = np.array(list(map(lambda x: [x[1], x[0], x[2]], point_one)))
        tp_2 = list(map(lambda x: [x[1], x[0], 1], self.right_manual_points))
        point_two = np.matmul(H2, np.array(tp_2).T)
        point_two /= point_two[2]
        point_two = point_two.T
        point_two = np.array(list(map(lambda x: [x[1], x[0], x[2]], point_two)))
        rectification = np.hstack((self.parameter_dict['Rectified_Params0'][0], self.parameter_dict['Rectified_Params1'][0]))
        for i in range(len(point_one)):
            cv.line(rectification, (int(point_one[i, 0]), int(point_one[i, 1])),
                     (int(point_two[i, 0]) + self.image_specs[0][1], int(point_two[i, 1])),
                     color=(0, 0, 255), thickness=2)
        cv.imwrite('Rectification.jpg', rectification)
        # return [rect_img1, rect_img2, F, point_one, point_two, H1, H2, P_dash]
        self.parameter_dict['P_dash_value'] = P_dash_value

    def rectify_image(self):

        for index, element in enumerate(self.parameter_dict['H1&H2']):
            correlation = self.get_correlation(index,element)
            values = [[min(correlation[0]), min(correlation[1])],[max(correlation[0]), max(correlation[1])]]
            d_im = np.array(values[1]) - np.array(values[0])
            d_im = [int(d_im[0]), int(d_im[1])]
            scale = np.array([[self.image_specs[index][1] / d_im[0], 0, 0], [0, self.image_specs[index][0] / d_im[1], 0], [0, 0, 1]])
            print(element.shape)
            H = np.matmul(scale, element)
            correlation = self.get_correlation(index,H)
            values_2 = [min(correlation[0]), min(correlation[1])]
            d_im = values_2
            d_im = [int(d_im[0]), int(d_im[1])]
            T_value = np.array([[1, 0, -1 * d_im[0] + 1], [0, 1, -1 * d_im[1] + 1], [0, 0, 1]], dtype=float)
            homography_n = np.matmul(T_value, H)
            inverse_homography = np.linalg.pinv(homography_n)
            result_image = self.create_image(index=index,H=inverse_homography)
            self.parameter_dict['Rectified_Params'+str(index)] = [result_image, homography_n]

    def ncc(self):
        corners_left = self.parameter_dict['corners_right']
        corners_right = self.parameter_dict['corners_right']
        image_left = self.image_pair[0]
        image_right = self.image_pair[1]
        ncc = np.zeros((len(corners_left), len(corners_right)))
        ncc = ncc - 2
        for row in range(len(corners_left)):
            for column in range(len(corners_right)):
                cor1 = corners_left[row];
                cor2 = corners_right[column]
                x_left_one = max(0, cor1[0] - self.padding)
                x_right_one = min(cor1[0] + self.padding + 1, image_left.shape[0])
                y_left_one = max(0, cor1[1] - self.padding)
                y_right_one = min(cor1[1] + self.padding + 1, image_left.shape[1])
                x_left_two = max(0, cor2[0] - self.padding)
                x_right_two = min(cor2[0] + self.padding + 1, image_right.shape[0])
                y_left_one = max(0, cor2[1] - self.padding)
                y_right_one = min(cor2[1] + self.padding + 1, image_right.shape[1])
                if x_right_one - x_left_one == x_right_two - x_left_two and y_right_one - y_left_one == y_right_one - y_left_one:
                    mean_value_one = np.mean(image_left[x_left_one:x_right_one, y_left_one:y_right_one])
                    mean_value_two = np.mean(image_right[x_left_two:x_right_two, y_left_one:y_right_one])
                    term_value_one = np.subtract(image_left[x_left_one:x_right_one, y_left_one:y_right_one], mean_value_one)
                    term_value_right = np.subtract(image_right[x_left_two:x_right_two, y_left_one:y_right_one], mean_value_two)
                    ncc[row, column] = np.divide(np.sum(np.multiply(term_value_one, term_value_right)),
                                                np.sqrt(
                                                    np.multiply(np.sum(np.square(term_value_one)), np.sum(np.square(term_value_right)))))

        to_ret = []
        nccs = []
        track = np.ones(len(corners_right))
        for row in range(len(ncc)):
            cur = ncc[row]
            to_find = cur[cur >= -1]
            if len(to_find) > 0:
                column = np.argmax(to_find)
                if abs(corners_left[row][0] - corners_right[column][0]) < 30 and abs(corners_left[row][1] - corners_right[column][1]) < 60 and track[
                    column] == 1:  # and max(to_find) > 0.45:
                    if track[column] == 1 and max(to_find) > 0.6:
                        nccs.append(max(to_find))
                        to_ret.append([corners_left[row], corners_right[column]])
                        track[column] = 0
        # sorted_idx = np.argsort(nccs)
        return (to_ret, nccs)

    def get_corners(self):
        edge_left = cv.Canny(self.grey_image_pair[0],25581.5,255)
        edge_right = cv.Canny(self.grey_image_pair[1],255*1.5,255)
        edge_left,edge_right = self.filter_edges([edge_left,edge_right])
        cv.imwrite('edges_left_image.jpg',edge_left)
        cv.imwrite('edges_right_image.jpg', edge_right)
        corners = list()
        for image in self.grey_image_pair:
            corner_list = list()
            count=0
            for row in range(image.shape[0]):
                for column in range(image.shape[1]):
                    if image[row, column] != 0:
                        count+=1
                        if count % 12 == 0:
                            corner_list.append([row,column])
            corners.append(corner_list)
        self.parameter_dict['corners_list_all'] = corners

    def filter_edges(self, edge_list):
        temp = []
        for element in edge_list:
            for row in range(element.shape[0]):
                for column in range(element.shape[1]):
                    if column < 40 or column > 350 or row < 100:
                        element[row, column] = 0
            temp.append(element)
        return temp

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
        elif type == 'H1':
            theta_value = self.get_theta(e_one=e_one,e_two=e_two,image_index=0, type='1')
            F_value = (np.cos(theta_value) * (e_one[0] - self.image_specs[0][1] / 2.0) - np.sin(theta_value) * (e_one[1] - self.image_specs[0][0] / 2.0))[0]
            R_value = np.array([[np.cos(theta_value)[0], -1 * np.sin(theta_value)[0], 0], [np.sin(theta_value)[0], np.cos(theta_value)[0], 0], [0, 0, 1]])
            T_value = np.array([[1, 0, -1 * self.image_specs[0][1] / 2.0], [0, 1, -1 * self.image_specs[0][0] / 2.0], [0, 0, 1]])
            G_value = np.array([[1, 0, 0], [0, 1, 0], [-1.0 / F_value, 0, 1]])
            H = np.matmul(np.matmul(G_value,R_value),T_value)
            assert H.shape == (3,3)
            print('Returning H1')
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

    def getROIFromUser(self, type = 'No'):
        """
        [This function is responsible for taking the regions of interests from the user for all the 4 pictures in order]

        """
        if type == 'yes':
            self.roiCoordinates = pickle.load(open('points.obj','rb'))
        else:
            self.roiList = []
            cv.namedWindow('Select ROI')
            cv.setMouseCallback('Select ROI', self.append_points)
            self.image = np.hstack((self.image_pair[0],self.image_pair[1]))
            while(True):
                cv.imshow('Select ROI', self.image)
                k = cv.waitKey(1) & 0xFF
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            pickle.dump(self.roiCoordinates,open('points.obj','wb'))
            cv.imwrite('result_1.jpg',self.image)


if __name__ == "__main__":
    tester = Reconstruct(['Task2_Images/Left.jpg','Task2_Images/Right.jpg'])
    tester.schedule()