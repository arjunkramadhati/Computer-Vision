"""
Computer Vision - Purdue University - Homework 5

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: September 28, 2020


[TO RUN CODE]: python3 imagemosaic.py
Output:
    [jpg]: Panoramic image stitched from 5 input images.
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
from scipy.optimize import least_squares
from scipy.optimize import minimize

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
        self.homographydict = {}
        self.kvalue = kvalue
        for i in range(len(self.image_addresses)):
            self.originalImages.append(cv.resize(cv.imread(self.image_addresses[i]), (320, 240)))
            self.grayscaleImages.append(cv.resize(cv.cvtColor(cv.imread(self.image_addresses[i]), cv.COLOR_BGR2GRAY), (320, 240)))
        self.siftobject = cv.SIFT_create()

    def weightedPixelValue(self, rangecoordinates, objectQueue):
        """
        [This function calculates the weighted pixel value at the given coordinate in the target image]

        Args:
            rangecoordinates ([list]): [This is the coordinate of the pixel in the target image]
            objectQueue ([int]): [This is the index number of the list which has the coordinates of the roI for the Object picture]

        Returns:
            [list]: [Weighted pixel value - RGB value]
        """

        pointOne = (int(np.floor(rangecoordinates[1])), int(np.floor(rangecoordinates[0])))
        pointTwo = (int(np.floor(rangecoordinates[1])), int(np.ceil(rangecoordinates[0])))
        pointThree = (int(np.ceil(rangecoordinates[1])), int(np.ceil(rangecoordinates[0])))
        pointFour = (int(np.ceil(rangecoordinates[1])), int(np.floor(rangecoordinates[0])))

        pixelValueAtOne = self.originalImages[objectQueue][pointOne[0]][pointOne[1]]
        pixelValueAtTwo = self.originalImages[objectQueue][pointTwo[0]][pointTwo[1]]
        pixelValueAtThree = self.originalImages[objectQueue][pointThree[0]][pointThree[1]]
        pixelValueAtFour = self.originalImages[objectQueue][pointFour[0]][pointFour[1]]

        weightAtOne = 1 / np.linalg.norm(pixelValueAtOne - rangecoordinates)
        weightAtTwo = 1 / np.linalg.norm(pixelValueAtTwo - rangecoordinates)
        weightAtThree = 1 / np.linalg.norm(pixelValueAtThree - rangecoordinates)
        weightAtFour = 1 / np.linalg.norm(pixelValueAtFour - rangecoordinates)

        return ((weightAtOne * pixelValueAtOne) + (weightAtTwo * pixelValueAtTwo) + (
                    weightAtThree * pixelValueAtThree) + (weightAtFour * pixelValueAtFour)) / (
                           weightAtFour + weightAtThree + weightAtTwo + weightAtOne)

    def get_panorama_done(self):
        """
        This function calls the necessary functions to get the final panorama image output
        :return: None
        """
        self.calculate_ransac_parameters()
        for i in range(0,4,1):
            self.perform_ransac((str(i),str(i+1),str(i)+str(i+1)))
        self.get_product_homography()
        self.get_panorama_image(('02','12','22','32','42'))

    def get_panorama_image(self,tags):
        """
        This function first computes the size of the final panorama image. This it stitches the 5 images into
        a panoramic image
        :param tags: Tags for key values where the respective homography matrices are stored in the dictionary
        :return: Stores the final image
        """
        cornerlist = []
        for i in range(len(tags)):
            endpoints = np.zeros((3,4))
            endpoints[:,0] = [0,0,1]
            endpoints[:,1] = [0, self.originalImages[i].shape[1], 1]
            endpoints[:,2] = [self.originalImages[i].shape[0],0, 1]
            endpoints[:,3] = [self.originalImages[i].shape[0],self.originalImages[i].shape[1], 1]
            corners = np.matmul(self.homographydict[tags[i]], endpoints)
            for i in range(corners.shape[1]):
                corners[:, i] = corners[:, i] / corners[-1, i]
            cornerlist.append(corners[0:2, :])
        minvalue =np.amin(np.amin(cornerlist,2),0)
        maxvalue = np.amax(np.amax(cornerlist, 2), 0)
        imagesize = maxvalue - minvalue
        pan_img = np.zeros((int(imagesize[1]), int(imagesize[0]), 3))
        for i in range(len(tags)):
            print(i)
            H = np.linalg.inv(self.homographydict[tags[i]])
            for column in range(0,pan_img.shape[0]):
                for row in range(0,pan_img.shape[1]):
                    print(str(column)+ " out of " + str(pan_img.shape[0]))
                    sourcecoord = np.array([row+minvalue[0], column+minvalue[1], 1])
                    destcoord = np.array(np.matmul(H,sourcecoord))
                    destcoord = destcoord/destcoord[-1]
                    if (destcoord[0]>0 and destcoord[1]>0 and destcoord[0]<self.originalImages[i].shape[1]-1 and destcoord[1]<self.originalImages[i].shape[0]-1):
                        pan_img[column][row] = self.weightedPixelValue(destcoord,i)

        cv.imwrite("panorama.jpg",pan_img)

    def get_product_homography(self):
        """
        Calculates the correct homography needed to get the final image. Since, we are taking the
        3rd image as the center image, we need all the homographies with respect to the 3rd image.
        :return: None. Stores the homographies in a dictionary
        """
        H02 = np.matmul(self.homographydict['01'], self.homographydict['12'])
        H02 = H02/H02[-1,-1]
        self.homographydict['02']=H02
        H12 = self.homographydict['12']/self.homographydict['12'][-1,-1]
        self.homographydict['12']=H12
        H32 = np.linalg.inv(self.homographydict['23'])
        H32 = H32/H32[-1,-1]
        self.homographydict['32']=H32
        H42=np.linalg.inv(np.matmul(self.homographydict['23'],self.homographydict['34']))
        H42=H42/H42[-1,-1]
        self.homographydict['42']=H42
        H22 = np.identity(3)
        self.homographydict['22']=H22

    def calculate_ransac_parameters(self, pvalue=0.999,epsilonvalue=0.40,samplesize=6 ):
        """
        Calculates the ransac parameters.
        :param pvalue: p value is taken as 99.9%
        :param epsilonvalue: We assume 40% of the correspondences are outliers
        :param samplesize: we take 6 random samples to calculate homography
        :return: None.
        """
        self.ransactrials = int((math.log(1-pvalue)/math.log(1-(1-epsilonvalue)**samplesize)))
        # self.ransaccutoffsize = int(math.ceil((1-epsilonvalue)*correspondencedatasize))

    def refine_homography_objective_function(self, H, sourcpoints, destinationpoints):
        """
        Objective function for the scipy optimise least squares function.
        :param H: Homography
        :param sourcpoints: The points in the correspondences which are in the source image
        :param destinationpoints: The points in the correspondences which are in the destination image
        :return: error between the predicted and the actual point in the destination image
        """
        H = H.reshape(3, 3)
        sourcpoints = np.concatenate((sourcpoints, np.ones((sourcpoints.shape[0], 1), np.float)), axis=1)
        predictedpoints = np.matmul(H, sourcpoints.T).T
        predictedpoints = predictedpoints // predictedpoints[:, 2].reshape(-1,1)
        error = (predictedpoints[:, :2] - destinationpoints) ** 2
        error = np.sqrt(np.sum(error, axis=1))

        return error

    def refine_homography(self, H, sourcepoints, destinationpoints):
        """
        Refines the homography using the scipy library
        :param H: Homography
        :param sourcpoints: The points in the correspondences which are in the source image
        :param destinationpoints: The points in the correspondences which are in the destination image
        :return: Refined homography matrix in 3X3 shape
        """
        refinedH = least_squares(self.refine_homography_objective_function, np.squeeze(H.reshape(-1, 1)), method='lm',
                                               args=(sourcepoints, destinationpoints))
        refinedH = refinedH.x.reshape(3, 3)
        return refinedH

    def calculate_lls_homography(self, image1points, image2points):
        """
        Function to calculate the homography using linear least squares mehtod
        :param image1points: The points in the correspondences which are in the source image
        :param image2points: The points in the correspondences which are in the destination image
        :return: Homography matrix in 3X3 shape
        """
        H = np.zeros((3, 3))
        # Setup the A Matrix
        A = np.zeros((len(image1points) * 2, 9))
        for i in range(len(image1points)):
            A[i * 2] = [0, 0, 0, -image1points[i, 0], -image1points[i, 1], -1, image2points[i, 1] * image1points[i, 0],
                        image2points[i, 1] * image1points[i, 1], image2points[i, 1]]
            A[i * 2 + 1] = [image1points[i, 0], image1points[i, 1], 1, 0, 0, 0, -image2points[i, 0] * image1points[i, 0],
                            -image2points[i, 0] * image1points[i, 1], -image2points[i, 0]]

        U, D, V = np.linalg.svd(A)
        V_T = np.transpose(V)
        H_elements = V_T[:, -1]
        H[0] = H_elements[0:3] / H_elements[-1]
        H[1] = H_elements[3:6] / H_elements[-1]
        H[2] = H_elements[6:9] / H_elements[-1]

        return H


    def perform_ransac(self,tags, samplesize=6, cutoff=3, refine =True):
        """
        Function to perform RANSAC to filter out inliers and outliers in the correspondences.
        :param tags: String values for the keys in the dictionaries being used to retrieve relevant data
        :param samplesize: 6 samples per trial
        :param cutoff: cut off value to decide inlier vs outlier
        :param refine: True if we need to refine homography, False if we do not need refinement
        :return: We call the draw function to draw the inliers and the outliers.
        """
        correspondence = self.correspondence[tags[2]]
        image1points = np.zeros((len(correspondence), 2))
        image2points = np.zeros((len(correspondence), 2))
        image1points = correspondence[:, 0:2]
        image2points = correspondence[:, 2:]
        count = 0
        listofinliersfinal =[]
        listofoutliersfinal = []
        homographyfinal =np.zeros((3,3))

        for iteration in range(self.ransactrials):
            print(str(iteration) + " of " + str(self.ransactrials))
            print(len(image1points))
            ip_index = np.random.randint(0, len(image1points), samplesize)
            image1sample = image1points[ip_index, :]
            image2sample = image2points[ip_index, :]
            H = self.calculate_lls_homography(image1sample, image2sample)
            dest_pts_estimate = np.zeros((image2points.shape), dtype='int')
            for index in range(len(image1points)):
                dest_pts_nonNorm = np.matmul(H, ([image1points[index, 0], image1points[index, 1], 1]))
                dest_pts_estimate[index, 0] = dest_pts_nonNorm[0] / dest_pts_nonNorm[-1]
                dest_pts_estimate[index, 1] = dest_pts_nonNorm[1] / dest_pts_nonNorm[-1]

            estimationerror = dest_pts_estimate - image2points
            errorsqaure = np.square(estimationerror)
            dist = np.sqrt(errorsqaure[:, 0] + errorsqaure[:, 1])
            validpointidx = np.where(dist <= cutoff)
            invalidpointidx = np.where(dist > cutoff)
            innlierlist=[]
            outlierlist =[]
            for i,element in enumerate(dist):
                if element <=cutoff:
                    innlierlist.append([image1points[i][1],image1points[i][0],dest_pts_estimate[i][1],dest_pts_estimate[i][0] ])
                else:
                    outlierlist.append([image1points[i][0], image1points[i][1], image2points[i][0], image2points[i][1]])

            Inliers = [1 for val in dist if (val < 3)]
            if len(Inliers) > count:
                count = len(Inliers)
                listofinliersfinal =innlierlist
                listofoutliersfinal =outlierlist
                homographyfinal = H

        if refine == True:
            print("Refining...")
            self.homographydict[tags[2]] = self.refine_homography(homographyfinal, image1points, image2points)
        else:
            self.homographydict[tags[2]]=homographyfinal
        print(len(listofinliersfinal))
        print(len(listofoutliersfinal))
        self.draw_inliers_outliers(tags, correspondence, homographyfinal, 3)

    def draw_inliers_outliers(self, tags, correspondences, homography, cutoffvalue):
        """
        We use this function to draw the inliers and the outliers on the image.
        :param tags: Values for the keys in the relevant dictionary
        :param correspondences: matched points
        :param homography: H matrix
        :param cutoffvalue: decision value to decide inliers and outliers
        :return: Writes the image with inliers and outliers
        """
        firstimage = self.originalImages[int(tags[0])]
        secondimage = self.originalImages[int(tags[1])]
        nrows = max(firstimage.shape[0], secondimage.shape[0])
        ncol = firstimage.shape[1] + secondimage.shape[1]
        resultimage = np.zeros((nrows, ncol, 3))
        resultimage[:firstimage.shape[0], :firstimage.shape[1]] = firstimage
        resultimage[:secondimage.shape[0], firstimage.shape[1]:firstimage.shape[1] + secondimage.shape[1]] = secondimage
        image1points = correspondences[:, 0:2]
        image2points = correspondences[:, 2:]
        inliersimage1 = []
        inliersimage2 = []
        for src_pt in range(len(image1points)):
            estimate = np.matmul(homography, [image1points[src_pt, 0], image1points[src_pt, 1], 1])
            estimate = estimate / estimate[-1]
            diff = estimate[0:2] - image2points[src_pt, :]
            errorinestimation = np.sqrt(np.sum(diff ** 2))
            if errorinestimation < cutoffvalue:
                inliersimage1.append(image1points[src_pt, :])
                inliersimage2.append(image2points[src_pt, :])
                cv.circle(resultimage, (int(image1points[src_pt, 0]), int(image1points[src_pt, 1])), 2, (255, 0, 0), 2)
                cv.circle(resultimage, (firstimage.shape[1] + int(image2points[src_pt, 0]), int(image2points[src_pt, 1])), 2,
                           (255, 0, 0), 2)
                cv.line(resultimage, (int(image1points[src_pt, 0]), int(image1points[src_pt, 1])),
                         (firstimage.shape[1] + int(image2points[src_pt, 0]), int(image2points[src_pt, 1])), (0, 255, 0))
            else:
                cv.circle(resultimage, (int(image1points[src_pt, 0]), int(image1points[src_pt, 1])), 2, (0, 0, 255), 2)
                cv.circle(resultimage, (firstimage.shape[1] + int(image2points[src_pt, 0]), int(image2points[src_pt, 1])), 2,
                           (0, 0, 255), 2)
                cv.line(resultimage, (int(image1points[src_pt, 0]), int(image1points[src_pt, 1])),
                         (firstimage.shape[1] + int(image2points[src_pt, 0]), int(image2points[src_pt, 1])), (0, 0, 255))

        cv.imwrite("results/"+str(tags[2])+"inlieroutlier.jpg", resultimage)

    def draw_correspondence_inlier_outlier(self, inlierlist,outlierlist):
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
        resultImage = np.hstack((self.originalImages[0], self.originalImages[1]))
        for element in inlierlist:
            print(element)
            p1x = int(element[1])
            p1y = int(element[0])
            p2x = int(element[1]) + 640
            p2y = int(element[0])
            cv.line(resultImage, (p1x,p1y), (p2x,p2y), [0, 255, 0], 1 )
            # cv.circle(resultImage,(columnvalueone, rowvalueone), 2, [0, 0, 0], 2)
            # cv.circle(resultImage, (columnvaluetwo, rowvaluetwo), 2, [0, 0, 0], 2)
        cv.imwrite("sdfdsfsdf.jpg",resultImage)

    def update_dict_values(self,tags):
        """
        Convert the matched type variables into the form that we need
        :param tags: Values for the keys in the dictionary
        :return: Stores the values in a dictionary
        """
        tempdict = dict()
        ip1=[]
        ip2=[]
        matchedpoints = self.correspondence[tags[2]]
        (keypoint1, descriptor1) = self.cornerpointdict[tags[0]]
        (keypoint2, descriptor2) = self.cornerpointdict[tags[1]]
        for matchedpoint in matchedpoints:
            imageoneindex = matchedpoint[0].queryIdx
            imagetwoindex = matchedpoint[0].trainIdx
            (x1, y1) = keypoint1[imageoneindex].pt
            (x2, y2) = keypoint2[imagetwoindex].pt
            # tempdict[(x1,y1)]=(x2,y2)
            ip1.append((x1,y1))
            ip2.append((x2,y2))
        ip1=np.array(ip1)
        ip2=np.array(ip2)
        x = np.concatenate((ip1,ip2),axis=1)
        self.correspondence[tags[2]] = x


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
            result = cv.drawMatchesKnn(self.originalImages[queueImages[0]], keypoint1, self.originalImages[queueImages[1]],keypoint2, filteredmatchedpoints,None, flags=2)
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

    """
    Code starts here
    """
    tester = Panorama(['input_images/1.jpg','input_images/2.jpg', 'input_images/3.jpg', 'input_images/4.jpg',
                       'input_images/5.jpg'], 0.707)
    for i in range(5):
        tester.sift_corner_detect(i, str(i))
    print("Detected SIFT interest points in 5 images.")
    for i in range(0,4,1):
        print(i)
        tester.sift_correpondence((i,i+1),(str(i),str(i+1),str(i)+str(i+1)), 'OpenCV')
        tester.update_dict_values((str(i),str(i+1),str(i)+str(i+1)))

    tester.get_panorama_done()
