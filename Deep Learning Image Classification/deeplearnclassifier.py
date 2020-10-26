"""
Computer Vision - Purdue University - Homework 8

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Oct 19, 2020


[TO RUN CODE]: python3 deeplearnclassifier.py
Output:
    [labels]: Predictions for the input images in the form of a confusion matrix.
"""

import cv2 as cv
import numpy as np
import pickle
import re
import glob
from sklearn import svm
from scipy import signal
from sklearn.model_selection import train_test_split

class Gramclassify:

    def __init__(self, trainingset_path, testingset_path, cvalue = 24, mvalue = 15, kvalue = 16):
        """
        Initialise the gram classify object with the parameters
        :param trainingset_path: Path to the training data set
        :param testingset_path: Path to the testing data set
        :param cvalue: Value for the number of channels for convolution
        :param mvalue: Value for the kernel size of mvalue X mvalue
        :param kvalue: Value for the kvalue X kvalue downsampling
        """
        np.random.seed(0)
        self.cvalue = cvalue
        self.mvalue = mvalue
        self.kvalue = kvalue
        self.operators = None
        self.pattern = re.compile("([a-zA-Z]+)([0-9]+)")
        self.training_path = glob.glob(trainingset_path)
        self.testing_path = glob.glob(testingset_path)
        self.training_path.remove('./imagesDatabaseHW8/training/rain141.jpg')
        self.training_path.remove('./imagesDatabaseHW8/training/shine131.jpg')
        self.training_images_path, self.validation_images_path = train_test_split(self.training_path, shuffle=True, test_size=0.25, random_state=0)
        self.prepare_convolutional_operators()
        print(f"Train image paths = {self.training_path}")
        print('Initialization complete')

    def get_label_string(self, element):
        """
        Since the data has just one directory where images of all the classes
        are present, we will need to mine for the label or the class name
        from the file name. This function returns the class or the label name
        from the given image path
        :param element: Image path
        :return: Return image label or class
        """
        return self.pattern.match(element.split('/')[-1].split('.')[0]).groups()[0]

    def downsample_vectorise(self, image):
        """
        Downsample the convolution output into kvalue X kvalue size.
        Next vectorise the downsampled array.
        :param image: Image channel to be downsampled
        :return: Returns the vector representation of the texture for that channel
        """
        return np.reshape(image[::self.kvalue, ::self.kvalue, :], (-1, self.cvalue))

    def prepare_convolutional_operators(self):
        """
        This function prepares the convolution operators which we will be
        using to convolve the image into C different channels.
        :return: Set the operator to the global operator value
        """
        operators = np.zeros((self.mvalue, self.mvalue, self.cvalue), np.float)
        for index in range(operators.shape[2]):
            operators[:, :, index] =  np.random.rand(self.mvalue, self.mvalue) * 2 - 1
            operators[:, :, index] -= np.mean(operators[:, :, index])
        self.operators = operators

    def generate_gram_matrix(self, image):
        """
        This function generates the gram matrix for the given image.
        :param image: Input image for which we need the gram matrix
        :return: Return the gram matrix
        """
        if len(image.shape) > 2:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        convolved_image = np.zeros((image.shape[0]- self.mvalue +1, image.shape[1] - self.mvalue + 1, self.cvalue), np.float)
        for channel in range(self.cvalue):
            convolved_image[:, :, channel] = signal.convolve(image, self.operators[:, :, channel], mode='valid')
        vector = self.downsample_vectorise(convolved_image)
        gram_matrix = np.matmul(vector.T, vector)
        gram_matrix = gram_matrix/ np.sum(gram_matrix)
        gram_matrix.reshape(1,-1)
        return gram_matrix.reshape(1, -1)

    def dump_data(self, classes, labels, grams):
        """
        We dump the data so that we need not train and generate the model every time we need to
        predict classes for new images.
        :param classes: List of the class names
        :param labels: List of the label names
        :param grams: List of all the gram matrices
        :return: enumerated label list used for prediction and also the dictionary needed for the same task
        """
        classdict = dict()
        enumerated_labels = []
        for index, element in enumerate(classes):
            classdict[element] = index
        for label in labels:
            enumerated_labels.append(classdict[label])
        pickle.dump(enumerated_labels, open('enumerated_labels.p', 'wb'))
        pickle.dump(classes, open('classes.p', 'wb'))
        pickle.dump(classdict, open('classdict.p', 'wb'))
        pickle.dump(labels, open('labels.p', 'wb'))
        pickle.dump(grams, open('grams.p', 'wb'))
        return enumerated_labels, classdict

    def construct_confusion_matrix(self, image_path, classes, dictionary, model):
        """
        We construct the confusion matrix given the image paths, classes
        and other necessary parameters
        :param image_path: Image paths
        :param classes: List of the classes
        :param dictionary: Dictionary of the classes and their indices
        :param model: SVM model needed to perform the predictions
        :return: Return the confusion matrix and the accuracy scores.
        """
        confusion_matrix = np.zeros((len(classes), len(classes)), np.float)
        for element in image_path:
            grayimage = cv.resize(cv.imread(element,0),(300,200))
            label = self.get_label_string(element)
            gram_matrix = self.generate_gram_matrix(grayimage)
            prediction = model.predict(gram_matrix)
            label_enumerate = dictionary[label]
            confusion_matrix[label_enumerate, prediction] += 1
        return confusion_matrix, np.trace(confusion_matrix)/np.sum(confusion_matrix)

    def construct_representational_space(self):
        """
        This function does the following:
        1) Constructs the C^2/2 dimensional representational space.
        2) Train the Support Vector Machine
        3) Validate the training
        4) Test the trained model by making predictions of new images
        :return: None. Prints the final result summary.
        """
        classes = []
        labels = []
        grams = np.zeros((len(self.training_images_path), self.cvalue*self.cvalue), np.float)
        for index, element in enumerate(self.training_images_path):
            print("Process complete: " + str(index/len(self.training_images_path)))
            grayimage = cv.resize(cv.imread(element,0),(300,200))
            label = self.get_label_string(element)
            if label not in classes:
                classes.append(label)
            gram_matrix = self.generate_gram_matrix(grayimage)
            grams[index] = gram_matrix
            labels.append(label)
        enumerated_labels, classdict = self.dump_data(classes, labels, grams)
        model = svm.SVC(kernel='poly')
        model.fit(grams, enumerated_labels)
        pickle.dump(model, open('model.p', 'wb'))
        confusion_matrix, accuracy = self.construct_confusion_matrix(self.validation_images_path,classes, classdict, model)
        print("SVM Training complete.")
        print("------------------------------------")
        print("------------------------------------")
        print('Validation complete...')
        print('Validation accuracy score: ' + str(accuracy* 100) + "%")
        print('Printing confusion matrix')
        print(confusion_matrix)
        confusion_matrix, accuracy = self.construct_confusion_matrix(self.testing_path, classes, classdict, model)
        print("------------------------------------")
        print("------------------------------------")
        print('Testing complete...')
        print('Testing accuracy score: ' + str(accuracy* 100) + "%")
        print('Printing confusion matrix')
        print(confusion_matrix)
        print("------------------------------------")
        print("------------------------------------")
        print("Parameter summary")
        print("C value for the number of channels: " + str(self.cvalue))
        print("M value for the kernel size: " + str(self.mvalue))
        print("K value for the downsampling: " + str(self.kvalue))
        print("------------------------------------")
        print("------------------------------------")

if __name__ == "__main__":
    """
    Code begins here
    """
    tester = Gramclassify('./imagesDatabaseHW8/training/*','./imagesDatabaseHW8/testing/*', )
    tester.construct_representational_space()