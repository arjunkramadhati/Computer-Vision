"""
Computer Vision - Purdue University - Homework 5

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: September 28, 2020


[TO RUN CODE]: python3 imagemosaic.py
Output:
    [jpg]: Panoramic image sticthed from 5 input images.
"""

import cv2 as cv
import math
import numpy as np
import time
from scipy import signal as sg
import tqdm
import copy
import threading

class Panorama:
    def __init__(self, image_addresses, scale, kvalue=0.04):
        self.image_addresses = image_addresses
        self.scale = scale
        self.originalImages = []
        self.grayscaleImages = []
        self.filters = {}
        self.cornerpointdict = {}
        self.slidingwindowdict = {}
        self.correspondence = {}
        self.kvalue = kvalue
        for i in range(len(self.image_addresses)):
            self.originalImages.append(cv.resize(cv.imread(self.image_addresses[i]), (640, 480)))
            self.grayscaleImages.append(cv.resize(cv.cvtColor(cv.imread(self.image_addresses[i]), cv.COLOR_BGR2GRAY), (640, 480)))
        self.siftobject = cv.SIFT_create()

    def calculate_ransac_parameters(self,correspondencedatasize, pvalue=0.99,epsilonvalue=0.20,samplesize=6 ):
        self.ransactrials = int(math.ceil(math.log(1-pvalue)/math.log(1-math.pow(1-epsilonvalue,samplesize))))
        self.ransaccutoffsize = int(math.ceil((1-epsilonvalue)*correspondencedatasize))

    def sift_corner_detect(self, queueImage, tag):
        """
        This function detected and computes the sift keypoints and the descriptors. We use the generated keypoints
        to draw them on the picture. These are the detected corners.
        :param queueImage: Index of the location at which the image under consideration is stored in the list
        :param tag: Values to access and store values by key in the dictionaries
        :return: None. Stores the image.
        """
        keypoint, descriptor = self.siftobject.detectAndCompute(self.grayscaleImages[queueImage], None)
        self.cornerpointdict[tag] = (keypoint, descriptor)
        img = cv.drawKeypoints(self.grayscaleImages[queueImage], keypoint, copy.deepcopy(self.originalImages[queueImage]), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite(tag + '.jpg', img)

    def sift_correpondence(self, queueImages, tags, method):
        """
        This function estimates the correspondences between the sift corners detected in the pair of images.
        We have two options to estimate this: 1) Using BFMatcher function of OpenCV to get K-Nearest
        Neighbors or 2) Use a custom built
        eucledian distance based estimator
        :param queueImages: Index of the location at which the image under consideration is stored in the list
        :param tags: Values to access and store values by key in the dictionaries
        :param method: Use BFMatcher or custom built eucledian matcher.
        :return: None. Stores the matched keypoints in a global dictionary self.correspondence
        """
        (keypoint1, descriptor1) = self.cornerpointdict[tags[0]]
        (keypoint2, descriptor2) = self.cornerpointdict[tags[1]]
        if method =='OpenCV':
            matchedpoints = cv.BFMatcher().knnMatch(descriptor1, descriptor2, k=2)
            filteredmatchedpoints = []
            for pointone, pointtwo in matchedpoints:
                if pointone.distance < (pointtwo.distance * 0.75):
                    filteredmatchedpoints.append([pointone])
            result = cv.drawMatchesKnn(self.grayscaleImages[queueImages[0]], keypoint1, self.grayscaleImages[queueImages[1]],keypoint2, filteredmatchedpoints,None, flags=2)
            cv.imwrite(str(queueImages[0]) + "Sift_Correspondence.jpg", result)
        elif method =='Custom':
            tempdict = dict()
            for index,element in enumerate(descriptor1):
                list = []
                list2 = []

                for index2, element2 in enumerate(descriptor2):
                    euclediandistance = np.sqrt(np.sum(np.square((element-element2))))
                    list.append(euclediandistance)
                    list2.append(keypoint2[index2])
                minimumvalue = min(list)
                id = list2[list.index(minimumvalue)]
                tempdict[(int(keypoint1[index].pt[1]),int(keypoint1[index].pt[0]))]=((int(id.pt[1]),int(id.pt[0])),minimumvalue)
            self.correspondence[tags[2]] = tempdict

if __name__ =='__main__':
    tester = Panorama(['input_images/1.jpg','input_images/2.jpg', 'input_images/3.jpg', 'input_images/4.jpg',
                       'input_images/5.jpg'], 0.707)
    for i in range(5):
        tester.sift_corner_detect(i, str(i))
    print("Detected SIFT interest points in 5 images.")
    for i in range(0,4,2):
        tester.sift_correpondence((i,i+1),(str(i),str(i+1)), 'OpenCV')

