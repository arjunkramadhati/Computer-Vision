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

    class LDA_Recog:

        def __init__(self, dataset_path, params):
            self.classes = params[0]
            self.samples = params[1]
            self.path = dataset_path
            self.params = dict()
            self.params['Vectors'] = list()
            self.params['Means'] = list()
            for index, element in enumerate(tqdm(self.path, desc='Data Loading')):
                folder = os.listdir(element)
                folder.sort()
                temp = list()
                for image_name in folder:
                    image_file = cv.imread(element+image_name)
                    image_file = cv.cvtColor(image_file, cv.COLOR_BGR2GRAY)
                    image_file = image_file.reshape((1,-1))
                    temp.append(image_file)
                mean, vector = self.process_data(vectors=temp, sizeX=len(folder))
                self.params['Vectors'].append(vector)
                self.params['Means'].append(mean)

        def scheduler(self):
            pass

        def get_mean_vector(self, vector):
            vec_difference = np.zeros(vector.shape)
            vec_mean = np.zeros((vector.shape[0], self.classes))
            for component in range(self.classes):
                vec_mean[:, component] = np.mean(vector[:, component * self.samples:(component + 1) * self.samples],
                                                 axis=1)
                vec_difference[:, component*self.samples:(component+1)*self.samples]=vector[:, component*self.samples:(component+1)*self.samples]-vec_mean[:,component, None]
            return vec_mean, vec_difference, (vec_mean-self.params['Means'][0])

        def get_label_vector(self):
            label_list = list()
            for class_number in range(self.classes):
                label_list.extend(np.ones((self.samples,1))[:,0]*(class_number+1))
            return np.asarray(label_list)

        def process_data(self, vectors, sizeX):
            vectors = (np.asarray(vectors).reshape((sizeX,-1))).transpose()
            vectors = vectors/np.linalg.norm(vectors, axis=0)
            value_mean = np.mean(vectors, axis=1)
            return value_mean, vectors

        def get_entry_value(self, d_value):
            return np.eye(self.classes)*(d_value**(-0.5))

        def compute(self):
            label_list = self.get_label_vector()
            mean, difference, mean_two = self.get_mean_vector(vector=self.params['Vectors'][0])
            d_value, u_value = np.linalg.eig(mean_two.transpose().dot(mean_two))
            index = np.argsort(-1*d_value)
            d_value = d_value[index]
            u_value = u_value[:, index]
            eigenvector = mean_two.dot(u_value)
            entry_value = self.get_entry_value(d_value=d_value)
            vectorZ = eigenvector.dot(entry_value)
            vectorX = np.dot(vectorZ.transpose(),difference)
            d_value_w, u_value_w = np.linalg.eig(vectorX.dot(vectorX.transpose()))
            index = np.argsort(d_value_w)
            u_value_w=u_value_w[:, index]



    class PCA_Reccog:
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
                print(element)
                for filename in os.listdir(element):
                    image_file = cv.imread(element+filename)
                    image_file = cv.cvtColor(image_file, cv.COLOR_BGR2GRAY)
                    image_file = image_file.flatten()
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
            W_matrix = W_matrix/np.linalg.norm(W_matrix, axis=0)
            return W_matrix

        def get_prediction(self, vector):
            return self.model_classify.predict(vector.T)

        def get_featurers(selfself, W_value, vector, component_number):
            return np.matmul(W_value[:, :component_number+1].T, vector)

        def get_results(self, guess):
            return (((guess==self.parameter_dict['Labels_Test']).sum())/630*100)

        def commence_analysis(self):
            value_Train, mu_value_Train, vector_Train = self.compute_mean(value=self.parameter_dict['Features_Train'].copy())
            error_value, error_vector = self.get_error_terms(vector=vector_Train)
            self.parameter_dict['Features_Train_Mod'] = value_Train
            value_Test, mu_value_Test, vector_Test = self.compute_mean(value=self.parameter_dict['Features_Test'].copy())
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
    # tester = RecognizeFace.PCA_Reccog(['ECE661_2020_hw11_DB1/train/','ECE661_2020_hw11_DB1/test/'], features=(630,16384), labels=630)
    # tester.scheduler()
    tester = RecognizeFace.LDA_Recog(['ECE661_2020_hw11_DB1/train/','ECE661_2020_hw11_DB1/test/'], params=(30,21))
