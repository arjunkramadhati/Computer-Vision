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
import scipy.signal as sc
from sklearn.neighbors import KNeighborsClassifier
from scipy.linalg import null_space
from scipy import optimize
import math
import random
import matplotlib.pyplot as plt

class RecognizeFace:

    def __init__(self, dataset_path, features, labels, neighbors=1):
        self.parameter_dict = dict()
        self.parameter_dict['Labels_Train'] = np.zeros((labels))
        self.parameter_dict['Labels_Test'] = np.zeros((labels))
        self.parameter_dict['Features_Train'] = np.zeros(features)
        self.parameter_dict['Features_Test'] = np.zeros(features)
        self.path = dataset_path
        self.model_classify = KNeighborsClassifier(n_neighbors=neighbors)

    def scheduler(self):
        self.prepare_data()
        self.commence_analysis()

    def add_features(self, idx, index, image_file):
        if idx == 0:
            self.parameter_dict['Features_Train'][index] = image_file
        elif idx == 1:
            self.parameter_dict['Features_Test'][index] = image_file

    def add_labels(self, idx, index, filename):
        if idx == 0:
            self.parameter_dict['Labels_Train'][index] = int(filename[0]+filename[1])
        elif idx == 0:
            self.parameter_dict['Labels_Test'][index] = int(filename[0]+filename[1])

    def prepare_data(self):

        for idx, element in enumerate(self.path):
            index = 0
            for filename in os.listdir(element):
                # image_file = cv.imread(self.path+filename)
                image_file = cv.cvtColor(cv.imread(element+filename), cv.COLOR_BGR2GRAY).flatten()
                self.add_features(idx=idx, index=index,image_file=image_file)
                self.add_labels(idx=idx, index=index, filename=filename)
                index+=1
        print('Data prep complete')

    def get_error_terms(self, vector):
        value, vector = np.linalg.eigh(np.matmul(vector.T,vector))
        index = np.argsort(-1*value)
        return value, vector[:, index]

    def get_W(self, vector, error_vector):
        W_matrix = np.matmul(vector, error_vector)
        return W_matrix/np.linalg.norm(W_matrix, axis=0)

    def get_prediction(self, vector):
        return self.model_classify.predict(vector.T)

    def get_featurers(selfself, W_value, vector, component_number):
        return np.matmul(W_value[:, :component_number+1].T, vector)

    def get_results(self, guess):
        return (((guess==self.parameter_dict['Labels_Test']).sum())/630*100)

    def commence_analysis(self):
        value_Train, mu_value_Train, vector_Train = self.compute_mean(value=self.parameter_dict['Features_Train'])
        error_value, error_vector = self.get_error_terms(vector=vector_Train)
        self.parameter_dict['Features_Train_Mod'] = value_Train
        value_Test, mu_value_Test, vector_Test = self.compute_mean(value=self.parameter_dict['Features_Test'])
        self.parameter_dict['Features_Test_Negated'] = value_Test
        W_value = self.get_W(vector=vector_Train, error_vector=error_vector)
        for component in range(30):
            trainingF = self.get_featurers(W_value=W_value, vector=vector_Train, component_number=component)
            testingF = self.get_featurers(W_value=W_value, vector=vector_Test, component_number=component)
            self.model_classify.fit(trainingF.T, self.parameter_dict['Labels_Train'])
            guess_value = self.get_prediction(vector=testingF)
            accuracy = self.get_results(guess=guess_value)
            print('PCA Face Rec complete. Score: '+str(accuracy))

    def compute_mean(self, value):
        value = np.transpose(value)
        value = value/np.linalg.norm(value, axis=0)
        mu_value = np.mean(value, axis=1)
        vector = value - mu_value[:, None]
        return value, mu_value, vector


if __name__ == "__main__":
    """
    Code starts here

    """
    tester = RecognizeFace(['ECE661_2020_hw11_DB1/train/','ECE661_2020_hw11_DB1/test/'], features=(630,16384), labels=630)
    tester.scheduler()