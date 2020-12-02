"""
------------------------------------
------------------------------------
Computer Vision - Purdue University - Homework 11
------------------------------------
Face Recognition & Object Detection
------------------------------------
Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Dec 2, 2020

Reference : https://engineering.purdue.edu/RVL/ECE661_2018/Homeworks/HW10/2BestSolutions/2.pdf
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
from sklearn.neighbors import KNeighborsClassifier


class RecognizeFace:

    class LDA_Recog:
        """
        This is the class for Face Recognition using LDA
        """
        def __init__(self, dataset_path, params, thresh = 30, neighbors = 1):
            """
            Initialise the LDA Face Rec object
            :param dataset_path: Path to two folders- Training, Testing
            :param params: Parameter list with class numbers and sample numbers
            :param thresh: Component cutoff
            :param neighbors: Number of neighbors needed from the KNN classifier
            """
            self.classes = params[0]
            self.samples = params[1]
            self.path = dataset_path
            self.params = dict()
            self.model_predictor = KNeighborsClassifier(n_neighbors=neighbors)
            self.thresh = thresh
            self.params['Vectors'] = list()
            self.params['Means'] = list()
            for index, element in enumerate(tqdm(self.path, desc='Data Loading')):
                folder = os.listdir(element)
                folder.sort()
                print(element)
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
            """
            This function handles all the functions in the right sequence to complete
            the task
            :return:
            """
            self.compute()

        def get_predictions(self, feature_set, label_set, test_set):
            """
            Fit the KNN model with the vector and get preditctions on the test set
            :param feature_set: Training feature vector
            :param label_set: Labels
            :param test_set: Test feature vector
            :return: precitions
            """
            feature_set = np.nan_to_num(feature_set)
            test_set = np.nan_to_num(test_set)
            self.model_predictor.fit(feature_set.transpose(), label_set)
            return self.model_predictor.predict(test_set.transpose())

        def get_transpose(self, ss):
            return ss.transpose()

        def get_acc(self, label_list, prediction):
            correct = np.zeros((len(label_list), 1))
            result = list()
            correct[prediction == label_list] = 1
            result += [np.sum(correct) / len(label_list)]
            return result

        def get_results(self, vectorZ, u_value_w, label_list):
            """
            This function returns the accuracy or the score of the model's
            prection capability
            :param vectorZ: Vector Z
            :param u_value_w: decomposed u value
            :param label_list: Labels
            :return: Accuracy
            """
            for value_K in range(self.thresh):
                ss = vectorZ.dot(u_value_w[: , :value_K+1])
                ss = ss/np.linalg.norm(ss, axis=0)
                value = self.get_transpose(ss=ss)
                trFeatures = np.dot(value, self.params['Vectors'][0]-self.params['Means'][0][:, None])
                tsFeatures = np.dot(value, self.params['Vectors'][1]-self.params['Means'][1][:, None])
                prediction = self.get_predictions(feature_set=trFeatures, label_set=label_list, test_set=tsFeatures)
                result = self.get_acc(label_list=label_list, prediction=prediction)
            return result

        def get_mean_vector(self, vector):
            """
            Function to get the mean vectors for LDA.
            :param vector: Initial image vector
            :return: Mean vector, difference vector and the vector B
            """
            vec_difference = np.zeros(vector.shape)
            vec_mean = np.zeros((vector.shape[0], self.classes))
            for component in range(self.classes):
                vec_mean[:, component] = np.mean(vector[:, component * self.samples:component * self.samples],
                                                 axis=1)
                vec_difference[:, component*self.samples+1:(component)*self.samples]=vector[:, component*self.samples+1:(component)*self.samples]-vec_mean[:,component, None]
            valLst = vec_mean-self.params['Means'][0][:, None]
            return vec_mean, vec_difference, valLst

        def get_label_vector(self):
            """
            Get label list
            :return: List of the labels
            """
            label_list = list()
            for class_number in range(self.classes):
                label_list.extend(np.ones((self.samples,1))[:,0]*(class_number+1))
            return np.asarray(label_list)

        def process_data(self, vectors, sizeX):
            """
            Get initial image vectors and their mean
            :param vectors: Image vector
            :param sizeX: Number of images in the directory
            :return: Mean, image vector
            """
            vectors = (np.asarray(vectors).reshape((sizeX,-1)))
            vectors = vectors.transpose()
            vectors = vectors/np.linalg.norm(vectors, axis=0)
            value_mean = np.mean(vectors, axis=1)
            return value_mean, vectors

        def get_sorted_du(self, d_value, u_value):
            index = np.argsort(-1 * d_value)
            d_value = d_value[index]
            u_value = u_value[:, index]
            return d_value, u_value

        def get_entry_value(self, d_value):
            return np.eye(self.classes)*(d_value**(-0.5))

        def compute(self):
            """
            This function computes and executes the face recognition
            using LDA.
            :return:
            """
            label_list = self.get_label_vector()
            mean, difference, mean_two = self.get_mean_vector(vector=self.params['Vectors'][0])
            d_value, u_value = np.linalg.eig(mean_two.transpose().dot(mean_two))
            d_value, u_value = self.get_sorted_du(d_value=d_value,u_value=u_value)
            eigenvector = mean_two.dot(u_value)
            entry_value = self.get_entry_value(d_value=d_value)
            vectorZ = eigenvector.dot(entry_value)
            vectorX = np.dot(vectorZ.transpose(),difference)
            vectorX = np.nan_to_num(vectorX)
            d_value_w, u_value_w = np.linalg.eig(vectorX.dot(vectorX.transpose()))
            _, u_value_w = self.get_sorted_du(d_value=d_value, u_value=u_value_w)
            result = self.get_results(vectorZ, u_value_w, label_list)
            file = open('result_lda.obj', 'wb')
            pickle.dump(result, file)
            print('LDA Face Recognition complete')

    class PCA_Reccog:
        """
        This is the class to perform Face Recognition using PCA
        """
        def __init__(self, dataset_path, features, labels, neighbors=4):
            """
            Initialise the object for PCA Face Recognition
            :param dataset_path: Path to the
            :param features: Size of feature vector
            :param labels: Size of label vector
            :param neighbors: Number of neighbors needed from the KNN predictor
            """
            self.parameter_dict = dict()
            self.parameter_dict['Labels_Train'] = np.zeros((labels))
            self.parameter_dict['Labels_Test'] = np.zeros((labels))
            self.parameter_dict['Features_Train'] = np.zeros(features)
            self.parameter_dict['Features_Test'] = np.zeros(features)
            self.path = dataset_path
            self.model_classify = KNeighborsClassifier(n_neighbors=neighbors)

        def scheduler(self):
            """
            This function runs all the required functions in the right order
            :return:
            """
            self.prepare_data()
            self.commence_analysis()

        def add_features(self, idx, index, image_file):
            """
            Build the features list for both training and testing data set
            :param idx: Identifier - 0 for train; 1 for test
            :param index: Image index
            :param image_file: Image read by opencv
            :return: None
            """
            if idx == 0:
                self.parameter_dict['Features_Train'][index] = image_file
            elif idx == 1:
                self.parameter_dict['Features_Test'][index] = image_file

        def add_labels(self, idx, index, filename):
            """
            Build the label list for both training and testing data set
            :param idx: Identifier - 0 for train; 1 for test
            :param index: Image index
            :param image_file: Image read by opencv
            :return: None
            """
            if idx == 0:
                self.parameter_dict['Labels_Train'][index] = int(filename[0]+filename[1])
            elif idx == 1:
                self.parameter_dict['Labels_Test'][index] = int(filename[0]+filename[1])

        def prepare_data(self):
            """
            This function prepares the data by reading, flattening and organizing all the images from
            both the datasets.
            :return: None
            """
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

        def get_eigen_terms(self, vector):
            """
            Get eigen terms
            :param vector: Image vector
            :return: eigen value and eigen vector
            """
            value, vector = np.linalg.eigh(np.matmul(vector.T, vector))
            index = np.argsort(-1*value)
            return value, vector[:, index]

        def get_W(self, vector, eigen_vector):
            """
            Get the W matrix
            :param vector: Image vector
            :param eigen_vector: Eigen vector
            :return: W matrix
            """
            W_matrix = np.matmul(vector, eigen_vector)
            W_matrix = W_matrix/np.linalg.norm(W_matrix, axis=0)
            return W_matrix

        def get_prediction(self, vector):
            """
            Get prediction from the model
            :param vector: Testing feature vector
            :return: Predictions
            """
            print(self.model_classify.predict(vector.T))
            return self.model_classify.predict(vector.T)

        def get_featurers(self, W_value, vector, component_number):
            return np.matmul(W_value[:, :component_number+1].T, vector)

        def get_results(self, guess):
            """
            Get the score for the model
            :param guess: Prediction
            :return: Accuracy
            """
            test_value = self.parameter_dict['Labels_Test']
            return ((guess==test_value).sum()/630*100)

        def commence_analysis(self):
            """
            This function executes the PCA Face Recognition pipeline
            :return: None
            """
            value_Train, mu_value_Train, vector_Train = self.compute_mean(value=self.parameter_dict['Features_Train'].copy())
            eigen_value, eigen_vector = self.get_eigen_terms(vector=vector_Train)
            self.parameter_dict['Features_Train_Mod'] = value_Train
            value_Test, mu_value_Test, vector_Test = self.compute_mean(value=self.parameter_dict['Features_Test'].copy())
            self.parameter_dict['Features_Test_Negated'] = value_Test
            W_value = self.get_W(vector=vector_Train, eigen_vector=eigen_vector)
            temp = list()
            for component in range(25):
                trainingF = self.get_featurers(W_value=W_value, vector=vector_Train, component_number=component)
                testingF = self.get_featurers(W_value=W_value, vector=vector_Test, component_number=component)
                self.model_classify.fit(trainingF.T, self.parameter_dict['Labels_Train'])
                guess_value = self.get_prediction(vector=testingF)
                accuracy = self.get_results(guess=guess_value)
                temp.append(accuracy)
                print(accuracy)
            print(temp)
            file = open('result_pca.obj', 'wb')
            pickle.dump(temp, file)

        def compute_mean(self, value):
            """
            Get the mean vectors
            :param value: Initial image file
            :return: modified image vector, mean value and the difference vector
            """
            value = np.transpose(value)
            value = value/np.linalg.norm(value, axis=0)
            mu_value = np.mean(value, axis=1)
            vector = value - mu_value[:, None]
            return value, mu_value, vector


if __name__ == "__main__":
    """
    Code starts here
    """
    tester = RecognizeFace.PCA_Reccog(['ECE661_2020_hw11_DB1/train/','ECE661_2020_hw11_DB1/test/'], features=(630,16384), labels=630)
    tester.scheduler()
    tester = RecognizeFace.LDA_Recog(['ECE661_2020_hw11_DB1/train/','ECE661_2020_hw11_DB1/test/'], params=(30,21))
    tester.scheduler()