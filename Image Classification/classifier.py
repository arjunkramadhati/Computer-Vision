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
import copy
import math
import glob
import BitVector
import pickle
import os
from collections import Counter

class Imageclassifier:

    def __init__(self, training_directory, testing_directory, parameterR, parameterP, kvalue, Train=False):
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
        if (delu < 0.001) and (delv < 0.001):
            interpolated_greylevel = float(image[centerX][centerY])
        elif (delv < 0.001):
            interpolated_greylevel = (1 - delu) * image[centerX][centerY] + delu * image[centerX + 1][centerY]
        elif (delu < 0.001):
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
        file = open(filename,'wb')
        pickle.dump(self.histogramdict, file)
        file.close()

    def load_data(self, filename):
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



    def knn_classify(self, lbp_hist_test_obj, nImgs = 5, nTrainImgs =20, nClass=5):
        k = self.kneighbors
        lbp_hist_train = self.database
        lbp_hist_test = np.zeros((nImgs, 10))
        for i in range(len(lbp_hist_test_obj)):  # Image number in test set
            lbp_hist_test[i, :] = np.array(list(lbp_hist_test_obj[i].values()))

        euclid_dt = np.zeros((nImgs, lbp_hist_train.shape[0]))
        test_img_labels = np.zeros((nImgs, k), dtype='int')
        label = np.zeros(nImgs, dtype='int')
        for i in range(nImgs):  # Number of images in test set
            for j in range(lbp_hist_train.shape[0]):  # Total images in training set
                euclid_dt[i, j] = np.linalg.norm(lbp_hist_test[i, :] - lbp_hist_train[j, 1:])
            euclid_dt_sort_idx = np.argsort(euclid_dt[i, :])
            euclid_dt_sort = np.sort(euclid_dt[i, :])
            # print(euclid_dt[i,:])
            # print(euclid_dt_sort_idx)

            for k_idx in range(k):
                if (euclid_dt_sort_idx[k_idx] < (nTrainImgs * 1)):
                    test_img_labels[i, k_idx] = 0
                elif (euclid_dt_sort_idx[k_idx] < (nTrainImgs * 2)):
                    test_img_labels[i, k_idx] = 1
                elif (euclid_dt_sort_idx[k_idx] < (nTrainImgs * 3)):
                    test_img_labels[i, k_idx] = 2
                elif (euclid_dt_sort_idx[k_idx] < (nTrainImgs * 4)):
                    test_img_labels[i, k_idx] = 3
                elif (euclid_dt_sort_idx[k_idx] < (nTrainImgs * 5)):
                    test_img_labels[i, k_idx] = 4

            # print(test_img_labels[i,:])
            # Get label with maximum appearence
            label[i], freq = Counter(list(test_img_labels[i, :])).most_common(1)[0]
            # print(label[i])
            # print(freq)

        return label

    def predict_and_analyse(self,blist,bblist,mlist,clist,tlist):

        label_index = self.knn_classify(blist)
        label_unique, label_unique_count = np.unique(label_index, return_counts=True)
        self.cmatrix['beach', label_unique] = label_unique_count
        label_index = self.knn_classify(bblist)
        label_unique, label_unique_count = np.unique(label_index, return_counts=True)
        self.cmatrix['building', label_unique] = label_unique_count
        label_index = self.knn_classify(mlist)
        label_unique, label_unique_count = np.unique(label_index, return_counts=True)
        self.cmatrix['mountain', label_unique] = label_unique_count
        label_index = self.knn_classify(clist)
        label_unique, label_unique_count = np.unique(label_index, return_counts=True)
        self.cmatrix['car', label_unique] = label_unique_count
        label_index = self.knn_classify(tlist)
        label_unique, label_unique_count = np.unique(label_index, return_counts=True)
        self.cmatrix['tree', label_unique] = label_unique_count
        print(self.cmatrix)


if __name__ == "__main__":
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
