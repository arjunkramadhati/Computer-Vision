"""
Computer Vision - Purdue University - Homework 7

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Oct 19, 2020


[TO RUN CODE]: python3 classifier.py
Output:
    [labels]: Predictions for the input images
"""

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math
import BitVector
import pickle
import os
from collections import Counter


class Imageclassifier:

    def __init__(self, training_directory, testing_directory, parameterR, parameterP, kvalue, Train=False):
        """
        Initialise the image classifier object used to either train or test the classification of images
        :param training_directory: Directory of the training set
        :param testing_directory: Directory of the testing set
        :param parameterR: Radius for the circular boundary
        :param parameterP: Number of points on the circular boundary
        :param kvalue:k nearest neighbors needed for match
        :param Train: Training yes or no
        """
        self.classdict = dict()
        self.imagedict = dict()
        self.histogramdict = dict()
        self.database = []
        self.cmatrix = np.zeros((5, 5), dtype='int')
        if not Train:
            training_directory = testing_directory
        self.classcount = len(os.listdir(training_directory))
        for element in os.listdir(training_directory):
            self.classdict[element] = len(os.listdir(training_directory+'/'+element))
            templist = []
            for image in sorted(os.listdir(training_directory+'/'+element)):
                print(os.listdir(training_directory+'/'+element))
                print(image)
                origimage = cv.imread(training_directory+'/'+element+'/'+image)
                imageread = np.zeros((origimage.shape[0], origimage.shape[1], origimage.shape[2]), dtype='uint8')
                image_gray = np.zeros((origimage.shape[0], origimage.shape[1]), dtype='uint8')
                imageread = cv.imread(training_directory+'/'+element+'/'+image)
                image_gray = cv.cvtColor(imageread, cv.COLOR_BGR2GRAY)
                templist.append(image_gray)
            self.imagedict[element]=templist
        self.parameterR = parameterR
        self.parameterP = parameterP
        self.kneighbors = kvalue
        print(self.imagedict['beach'][0].shape)

    def get_pixel_value(self, queuetuple, delu, delv, centerX, centerY):
        """
        This function implements the bilinear interpolation method used to get the pixel value
        or the grey value at the point p
        :param queuetuple: Location of the image
        :param delu: change in value in x direction
        :param delv: change in value in y direction
        :param centerX: the point at the center of the circle under consideration
        :param centerY: the point at the center of the circle under consideration
        :return: greylevel at the point pß
        """
        image = self.imagedict[queuetuple[0]][queuetuple[1]]
        if (delu < 0.01) and (delv < 0.01):
            interpolated_greylevel = float(image[centerX][centerY])
        elif (delv < 0.01):
            interpolated_greylevel = (1 - delu) * image[centerX][centerY] + delu * image[centerX + 1][centerY]
        elif (delu < 0.01):
            interpolated_greylevel = (1 - delv) * image[centerX][centerY] + delv * image[centerX][centerY + 1]
        else:
            interpolated_greylevel = (1 - delu) * (1 - delv) * image[centerX][centerY] + (1 - delu) * delv * image[centerX][centerY + 1] + delu * delv * \
                          image[centerX + 1][centerY + 1] + delu * (1 - delv) * image[centerX + 1][centerY]
        return interpolated_greylevel

    def build_histogram(self, histogram, runs):
        if len(runs) > 2:
            histogram[self.parameterP + 1] += 1
        elif len(runs) == 1 and runs[0][0] == '1':
            histogram[self.parameterP] += 1
        elif len(runs) == 1 and runs[0][0] == '0':
            histogram[0] += 1
        else:
            histogram[len(runs[1])] += 1
        return histogram

    def generate_texture_feature(self, queuetuple, Train = False):
        """
        This function implements the building of the Local Binary Pattern histogram for the give image
        :param queuetuple: Location of the image in the dictionary
        :return: None. Stores the histogram in a dictionary
        """
        histogram = {bins: 0 for bins in range(self.parameterP + 2)}
        greyimage = self.imagedict[queuetuple[0]][queuetuple[1]]
        for row in range(self.parameterR, greyimage.shape[0]-self.parameterR-1):
            # print(str(row) + " out of " + str(greyimage.shape[0] - self.parameterR - 1))
            for column in range(self.parameterR, greyimage.shape[1]- self.parameterR-1):
                binarypatternforpoint = []
                for pointnumber in range(self.parameterP):
                    delu = self.parameterR * math.cos(2 * math.pi * pointnumber / self.parameterP)
                    delv = self.parameterR * math.sin(2 * math.pi * pointnumber / self.parameterP)
                    if abs(delu) < 0.001: delu = 0.0
                    if abs(delv) < 0.001: delv = 0.0
                    greylevel = self.get_pixel_value(queuetuple, delu, delv, int(row+delu), int(column+delv))
                    if greylevel >= greyimage[row][column]:
                        binarypatternforpoint.append(1)
                    else:
                        binarypatternforpoint.append(0)
                bitvector = BitVector.BitVector(bitlist=binarypatternforpoint)
                intvals_for_circular_shifts = [int(bitvector << 1) for _ in range(self.parameterP)]
                minimum_bit_vector = BitVector.BitVector(intVal=min(intvals_for_circular_shifts), size=self.parameterP)
                runs = minimum_bit_vector.runs()
                histogram = self.build_histogram(histogram, runs)
        if Train:
            self.histogramdict[queuetuple] = histogram
            plt.bar(list(histogram.keys()), histogram.values(), color='b')
            path = 'histograms/' + str(queuetuple[0]) + '/'
            plt.savefig(path + 'Class_{}'.format(queuetuple[0]) + '_ImageNum_{}'.format(int(queuetuple[1])) + '.png')
        if not Train:
            return histogram

    def save_histograms_of_all(self, filename):
        """
        Saves the Local Binary Pattern histograms of every image
        :param filename: File name for the data base
        :return: None
        """
        file = open(filename,'wb')
        pickle.dump(self.histogramdict, file)
        file.close()

    def load_data(self, filename):
        """
        Loads the database from the saved .obj file. The database contains
        the LBP histograms of very image in training set
        :param filename: File name of the database being retrieved.
        :return: Load and store the database in a dictionary
        """
        blist =[]
        bblist =[]
        clist =[]
        mlist =[]
        tlist =[]
        beachlist =np.zeros((20,10))
        buildinglist =np.zeros((20,10))
        carlist =np.zeros((20,10))
        mountainlist =np.zeros((20,10))
        treelist=np.zeros((20,10))
        print(self.classdict['tree'])
        file = open(filename, 'rb')
        database = pickle.load(file)
        file.close()
        for element,index in database:
            if element[0] =='beach':
                blist.append(database.get(element))
            if element[0] =='building':
                bblist.append(database.get(element))
            if element[0] =='mountain':
                mlist.append(database.get(element))
            if element[0] =='car':
                clist.append(database.get(element))
            if element[0] =='tree':
                tlist.append(database.get(element))

        for index in range(len(blist)):
            beachlist[index, :] = np.array(list(blist[index].values()))
        for index in range(len(bblist)):
            buildinglist[index, :] = np.array(list(bblist[index].values()))
        for index in range(len(clist)):
            carlist[index, :] = np.array(list(clist[index].values()))
        for index in range(len(mlist)):
            mountainlist[index, :] = np.array(list(mlist[index].values()))
        for index in range(len(tlist)):
            treelist[index, :] = np.array(list(tlist[index].values()))

        histogram_all = np.zeros((100, 11))
        for i in range(5):
            index1 = 20 * i
            index2 = index1 + 20
            histogram_all[index1:index2, 0] = i
        histogram_all[:, 1:] = np.concatenate((beachlist, buildinglist, carlist, mountainlist, treelist), axis=0)
        self.database = histogram_all
        print('loaded data successfully')

    def knn_classify(self, list_histograms_class, numberoftestingimages = 5, numberoftrainingimages =20):
        """
        This function implements the knnparameter-Nearest Neighbor algorithm. We use the Eucledian distance to
        calculate the nearest matches.
        :param list_histograms_class: List of all the LBP histograms of a particular class
        :param numberoftestingimages: Number of images in the testing set
        :param numberoftrainingimages: Number of images in the training set
        :param nClass: Number of classes to predict
        :return: returns the index of the label of the classes
        """
        knnparameter = self.kneighbors
        training_histogra_all = self.database
        result_hist = np.zeros((numberoftestingimages, 10))
        condition1 = numberoftrainingimages * 1
        condition2 = numberoftrainingimages * 2
        condition3 = numberoftrainingimages * 3
        condition4 = numberoftrainingimages * 4
        condition5 = numberoftrainingimages * 5

        for i in range(len(list_histograms_class)):
            result_hist[i, :] = np.array(list(list_histograms_class[i].values()))
        eucledian_distance = np.zeros((numberoftestingimages, training_histogra_all.shape[0]))
        label_list = np.zeros((numberoftestingimages, knnparameter), dtype='int')
        labelindex = np.zeros(numberoftestingimages, dtype='int')
        for i in range(numberoftestingimages):
            for j in range(training_histogra_all.shape[0]):
                eucledian_distance[i, j] = np.linalg.norm(result_hist[i, :] - training_histogra_all[j, 1:])
            sorted_distance = np.argsort(eucledian_distance[i, :])
            for k_idx in range(knnparameter):
                if (sorted_distance[k_idx] < (condition1)):
                    label_list[i, k_idx] = 0
                elif (sorted_distance[k_idx] < (condition2)):
                    label_list[i, k_idx] = 1
                elif (sorted_distance[k_idx] < (condition3)):
                    label_list[i, k_idx] = 2
                elif (sorted_distance[k_idx] < (condition4)):
                    label_list[i, k_idx] = 3
                elif (sorted_distance[k_idx] < (condition5)):
                    label_list[i, k_idx] = 4
            labelindex[i], freq = Counter(list(label_list[i, :])).most_common(1)[0]
        return labelindex

    def predict_and_analyse(self,blist,bblist,mlist,clist,tlist):
        """
        This function takes the histograms of the testing set and uses them to
        predict the class labels for each of the images in the testing set. The results are
        collated in a confusion matrix
        :param blist: List of histograms for beach class
        :param bblist: List of histograms for building class
        :param mlist: List of histograms for mountain class
        :param clist: List of histograms for the car class
        :param tlist: List of histograms for the tree class
        :return: Prints the final confusion matrix.
        """

        label_index = self.knn_classify(blist)
        label_unique, label_unique_count = np.unique(label_index, return_counts=True)
        self.cmatrix[0, label_unique] = label_unique_count
        label_index = self.knn_classify(bblist)
        label_unique, label_unique_count = np.unique(label_index, return_counts=True)
        self.cmatrix[1, label_unique] = label_unique_count
        label_index = self.knn_classify(mlist)
        label_unique, label_unique_count = np.unique(label_index, return_counts=True)
        self.cmatrix[2, label_unique] = label_unique_count
        label_index = self.knn_classify(clist)
        label_unique, label_unique_count = np.unique(label_index, return_counts=True)
        self.cmatrix[3, label_unique] = label_unique_count
        label_index = self.knn_classify(tlist)
        label_unique, label_unique_count = np.unique(label_index, return_counts=True)
        self.cmatrix[4, label_unique] = label_unique_count
        print('Prediction complete')
        print('Printing confusion matrix...')
        print(self.cmatrix)


if __name__ == "__main__":
    """
    Code begins here
    """
    tester = Imageclassifier("imagesDatabaseHW7/training", "imagesDatabaseHW7/testing", 1, 8, 5)
    Train = False
    if Train:
        for element in os.listdir("imagesDatabaseHW7/training"):
            for index in range(len(os.listdir("imagesDatabaseHW7/training" + '/' + element))):
                print('training image class: '+element+' __ ' + str(index))
                tester.generate_texture_feature((element, index))
        print('Training complete. Saving histogram dictionary...')
        tester.save_histograms_of_all('histograms.obj')
        print('Saving complete')
    tester.load_data('histograms.obj')
    btestlist =[]
    bbtestlist =[]
    mtestlist = []
    ctestlist = []
    ttestlist = []
    testdict = dict()
    for element in os.listdir("imagesDatabaseHW7/testing"):
        for index in range(len(os.listdir("imagesDatabaseHW7/testing" + '/' + element))):
            print('testing image class: ' + element + ' __ ' + str(index))
            hist = tester.generate_texture_feature((element,index))
            if element == 'beach':
                btestlist.append(hist)
            if element == 'building':
                bbtestlist.append(hist)
            if element == 'mountain':
                mtestlist.append(hist)
            if element == 'car':
                ctestlist.append(hist)
            if element == 'tree':
                ttestlist.append(hist)
    tester.predict_and_analyse(btestlist,bbtestlist,mtestlist,ctestlist,ttestlist)

