"""
------------------------------------
------------------------------------
Computer Vision - Purdue University - Homework 11
------------------------------------
Face Recognition & Object Detection
------------------------------------
Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Dec 2, 2020
------------------------------------
[TO RUN CODE]: python3 source_code.py
------------------------------------
------------------------------------
"""
import os
import pickle
import cv2 as cv
import numpy as np
from tqdm import tqdm
from scipy import spatial
from scipy.linalg import null_space
from scipy import optimize
import matplotlib.pyplot as plt

class FaceRecognition:
    def __init__(self, file_locations, classes, samples, components):
        self.testing_path = os.listdir(file_locations[0])
        self.training_path = os.listdir(file_locations[1])
        self.paths = file_locations
        self.parameter_dict = dict()
        self.parameter_dict['Classes'] = classes
        self.parameter_dict['Samples'] = samples
        self.parameter_dict['PrincipleComponents'] = components

    def process_results(self):
        self.get_pca_performance()

    def get_image_mean(self, images):
        return np.mean(images, axis=0)

    def get_eigen_vectors(self, VectorX):
        eigen_value, eigen_vector = np.linalg.eig(np.matmul(VectorX.T, VectorX))
        return ((eigen_vector[np.argsort(eigen_value)[::-1]]).T).T

    def construct_X(self,images, mean):
        VectorX = (images-mean).T
        for column in range(VectorX.shape[1]):
            VectorX[:, column] = VectorX[:, column] / np.linalg.norm(VectorX[:, column])
        return VectorX

    def construct_W(self, VectorX, eigen_vector, component):
        W_matrix = np.matmul(VectorX, eigen_vector)
        for column in range(W_matrix.shape[1]):
            W_matrix[:, column] = W_matrix[:, column]/np.linalg.norm(W_matrix[:, column])
        return W_matrix, W_matrix[:, :component]

    def get_minimum_distance(self, distance):
        return np.argmin(distance)

    def get_accuracy(self,test, train):
        accuracy = [0]*self.parameter_dict['Classes']
        for column in range(test.shape[1]):
            distance = list()
            for row in range(train.shape[1]):
                distance.append(spatial.distance.euclidean(test[:,column],train[:,row]))
            minimum = self.get_minimum_distance(distance=distance)
            if int(minimum/self.parameter_dict['Samples']) == column:
                accuracy[column] +=1
        return accuracy

    def get_pca_performance(self):
        for component in tqdm(range(self.parameter_dict['PrincipleComponents']), desc='Component Analysis'):
            array_images = list()
            for image_name in self.training_path:
                array_images.append(cv.cvtColor(cv.imread(self.paths[1]+'/'+image_name), cv.COLOR_BGR2GRAY).flatten())
            assert len(array_images) == 630
            array_images = np.array(array_images, dtype=np.float32)
            mean_value = self.get_image_mean(images=array_images)
            VectorX = self.construct_X(images=array_images, mean=mean_value)
            eigen_vector = self.get_eigen_vectors(VectorX=VectorX)
            W_matrix, largest_value = self.construct_W(VectorX=VectorX, eigen_vector=eigen_vector, component=component)
            training_y = np.matmul(largest_value.T, VectorX)
            array_images = list()
            for image_name in self.testing_path:
                array_images.append(cv.cvtColor(cv.imread(self.paths[0]+'/'+image_name), cv.COLOR_BGR2GRAY).flatten())
            assert len(array_images) == 630
            array_images = np.array(array_images, dtype=np.float32)
            ################################
            #Check this area for results
            ################################
            mean_value = self.get_image_mean(images=array_images)
            VectorX = self.construct_X(images=array_images,mean=mean_value)
            testing_y = np.matmul(largest_value.T, VectorX)
            accuracy = self.get_accuracy(test=testing_y, train=training_y)
            print(accuracy)










if __name__ == "__main__":
    """
    Code starts here

    """
    tester = FaceRecognition(['ECE661_2020_hw11_DB1/test','ECE661_2020_hw11_DB1/train'], classes=30, samples=21,components=15)
    tester.process_results()