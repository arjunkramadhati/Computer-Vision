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
import random
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

    def get_panorama_done(self,tag):
        correspondencedatasize = len(list(self.correspondence[tag].keys()))
        self.calculate_ransac_parameters(correspondencedatasize)
        self.perform_ransac(tag)

    def calculate_ransac_parameters(self,correspondencedatasize, pvalue=0.99,epsilonvalue=0.20,samplesize=6 ):
        self.ransactrials = int(math.ceil(math.log(1-pvalue)/math.log(1-math.pow(1-epsilonvalue,samplesize))))
        self.ransaccutoffsize = int(math.ceil((1-epsilonvalue)*correspondencedatasize))

    def calculate_lls_homography(self,points, samplesize=6):
        homography = np.zeros((3,3))
        amatrix = np.zeros((2*samplesize,9))

        for index in range(samplesize):
            amatrix[2*index] = [0,0,0,-points[i][0][0],-points[i][0][1],-1,points[i][1][1]*points[i][0][0],
                                points[i][1][1]*points[i][0][1], points[i][1][1]]
            amatrix[2*index +1] = [points[i][0][0],points[i][0][1],1,0,0,0,-points[i][1][0]*points[i][0][0],
                                   -points[i][1][0]*points[i][0][1],-points[i][1][0]]

        uvalue, dvalue, vvalue = np.linalg.svd(amatrix)
        vvalueT = np.transpose(vvalue)
        solution = vvalueT[:,-1]
        homography[0] = solution[0:3]/solution[-1]
        homography[1] = solution[3:6]/solution[-1]
        homography[2] = solution[6:9]/solution[-1]
        return homography

    def perform_ransac(self,tag, samplesize=6, cutoff=3):
        correspondence = self.correspondence[tag]
        sourcepoints = []
        destinationpoints = []
        sx = list(correspondence.keys())
        dx = list(correspondence.values())
        for key,value in correspondence.items():
            sourcepoints.append(key[0])
            sourcepoints.append(key[1])
            sourcepoints.append(1.0)
            destinationpoints.append(value[0])
            destinationpoints.append(value[1])
            destinationpoints.append(1.0)
        sourcepoints = np.array(sourcepoints, dtype='float64')
        sourcepoints = sourcepoints.reshape(-1,3).T
        destinationpoints = np.array(destinationpoints, dtype='float64')
        destinationpoints = destinationpoints.reshape(-1,3).T
        count = 0
        listofinliersfinal =[]
        homographyfinal =np.zeros((3,3))

        for iteration in range(self.ransactrials):
            print(str(iteration) + " of " + str(self.ransactrials))
            samples =random.sample(list(correspondence.items()),samplesize)
            estimatehomography = self.calculate_lls_homography(samples)
            estimatedpoints = np.matmul(estimatehomography,sourcepoints)
            # print(estimatedpoints)
            estimatedpoints = estimatedpoints/estimatedpoints[2,:]
            squaredifference = (estimatedpoints - destinationpoints)**2
            sumdifference =np.sum(squaredifference, axis=0)
            validpointsidx = np.where(sumdifference <= cutoff**2)
            print(validpointsidx[0])
            listofinliersleft = [sx[i] for i in validpointsidx[0]]
            if len(listofinliersleft) > count:
                count = len(listofinliersleft)
                listofinliersfinal = listofinliersleft
                homographyfinal = estimatehomography
        print(listofinliersfinal)



    def update_dict_values(self,tags):
        tempdict = dict()
        matchedpoints = self.correspondence[tags[2]]
        (keypoint1, descriptor1) = self.cornerpointdict[tags[0]]
        (keypoint2, descriptor2) = self.cornerpointdict[tags[1]]
        for matchedpoint in matchedpoints:
            imageoneindex = matchedpoint[0].queryIdx
            imagetwoindex = matchedpoint[0].trainIdx
            (x1, y1) = keypoint1[imageoneindex].pt
            (x2, y2) = keypoint2[imagetwoindex].pt
            tempdict[(x1,y1)]=(x2,y2)
        self.correspondence[tags[2]] = tempdict
        print(self.correspondence[tags[2]])


    def draw_correspondence(self, tags, cutoffvalue, style):
        """
        This function draws the correspondence between the corner points in the pair of images. We denote each
        corner point by a small circle around it. The correspondence is denoted by a line connecting the two points.
        Before drawing the points, we first filter the correspondences based on a cutoff value so that we retain only
        fairly accurate matches and not completely-off matches.
        :param tags: Values to access and store values by key in the dictionaries
        :param cutoffvalue: Value used to filter the list of matched corner points
        :param style: Either filter values above the cutoff value or filter the values below it.
        :return: Returns the resultant stitched image with the denoted correspondence lines.
        """
        copydict = copy.deepcopy(self.correspondence[tags[0]])
        print(copydict)
        for (key,value) in self.correspondence[tags[0]].items():
            if style == 'greaterthan':
                if value[1]> cutoffvalue:
                    copydict.pop(key)
            elif style == 'lesserthan':
                if value[1]< cutoffvalue:
                    copydict.pop(key)
        resultImage = np.hstack((self.originalImages[0], self.originalImages[1]))
        horizontaloffset = 640
        print(copydict)
        for (key,value) in copydict.items():
            # print((key,value))
            columnvalueone = key[1]
            rowvalueone = key[0]
            columnvaluetwo = value[0][1] + horizontaloffset
            rowvaluetwo = value[0][0]
            cv.line(resultImage, (columnvalueone,rowvalueone), (columnvaluetwo,rowvaluetwo), [0, 255, 0], 1 )
            cv.circle(resultImage,(columnvalueone, rowvalueone), 2, [0, 0, 0], 2)
            cv.circle(resultImage, (columnvaluetwo, rowvaluetwo), 2, [0, 0, 0], 2)
        return resultImage

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
        #img = cv.drawKeypoints(self.grayscaleImages[queueImage], keypoint, copy.deepcopy(self.originalImages[queueImage]), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv.imwrite(tag + '.jpg', img)

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
            #print(matchedpoints)
            filteredmatchedpoints = []
            for pointone, pointtwo in matchedpoints:
                if pointone.distance < (pointtwo.distance * 0.75):
                    filteredmatchedpoints.append([pointone])
            self.correspondence[tags[2]]=filteredmatchedpoints
            result = cv.drawMatchesKnn(self.grayscaleImages[queueImages[0]], keypoint1, self.grayscaleImages[queueImages[1]],keypoint2, filteredmatchedpoints,None, flags=2)
            cv.imwrite("results/"+ str(tags[2]) + ".jpg", result)
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
    for i in range(0,4,1):
        print(i)
        tester.sift_correpondence((i,i+1),(str(i),str(i+1),str(i)+str(i+1)), 'OpenCV')
        tester.update_dict_values((str(i),str(i+1),str(i)+str(i+1)))
        # image = tester.draw_correspondence((str(i)+str(i+1),'value'),650,'greaterthan')
        # cv.imwrite(str(i)+str(i+1)+'.jpg', image)
    tester.get_panorama_done('01')
