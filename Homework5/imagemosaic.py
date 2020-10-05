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
            self.originalImages.append(cv.resize(cv.imread(self.image_addresses[i]), (640, 480)))
            self.grayscaleImages.append(cv.resize(cv.cvtColor(cv.imread(self.image_addresses[i]), cv.COLOR_BGR2GRAY), (640, 480)))
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
        #correspondencedatasize = len(list(self.correspondence[tag].keys()))
        self.calculate_ransac_parameters()
        for i in range(0,4,1):
            self.perform_ransac((str(i),str(i+1),str(i)+str(i+1)))
        self.get_product_homography()
        self.get_panorama_image_size(('02','12','22','32','42'))

    def get_panorama_image_size(self,tags):
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
                    sourcecoord = np.array([row+minvalue[0], column+minvalue[1], 1])
                    destcoord = np.array(np.matmul(H,sourcecoord))
                    destcoord = destcoord/destcoord[-1]
                    if (destcoord[0]>0 and destcoord[1]>0 and destcoord[0]<self.originalImages[i].shape[1]-1 and destcoord[1]<self.originalImages[i].shape[0]-1):
                        pan_img[column][row] = self.weightedPixelValue(destcoord,i)

        cv.imwrite("panorama.jpg",pan_img)

    def get_product_homography(self):
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
        self.ransactrials = int((math.log(1-pvalue)/math.log(1-(1-epsilonvalue)**samplesize)))
        # self.ransaccutoffsize = int(math.ceil((1-epsilonvalue)*correspondencedatasize))

    def LinearLeastSquaresHomography(self,src_pts, dest_pts):
        # Initialize Homography Matrix
        H = np.zeros((3, 3))
        # Setup the A Matrix
        A = np.zeros((len(src_pts) * 2, 9))
        for i in range(len(src_pts)):
            A[i * 2] = [0, 0, 0, -src_pts[i, 0], -src_pts[i, 1], -1, dest_pts[i, 1] * src_pts[i, 0],
                        dest_pts[i, 1] * src_pts[i, 1], dest_pts[i, 1]]
            A[i * 2 + 1] = [src_pts[i, 0], src_pts[i, 1], 1, 0, 0, 0, -dest_pts[i, 0] * src_pts[i, 0],
                            -dest_pts[i, 0] * src_pts[i, 1], -dest_pts[i, 0]]

        # print(A)
        # print("done")
        # Do SVD Decomposition
        U, D, V = np.linalg.svd(A)
        V_T = np.transpose(V)  # Need to take transpose because rows of V are eigen vectors
        H_elements = V_T[:, -1]  # Last column is the solution

        # Fill the Homography Matrix
        H[0] = H_elements[0:3] / H_elements[-1]
        H[1] = H_elements[3:6] / H_elements[-1]
        H[2] = H_elements[6:9] / H_elements[-1]
        # H = np.linalg.pinv(H)
        # print(H)
        # print("done")
        return H
    # def calculate_lls_homography(self,points, samplesize=6):
    #     homography = np.zeros((3,3))
    #     amatrix = np.zeros((2*samplesize,9))
    #     for index in range(samplesize):
    #         amatrix[2*index] = [0,0,0,-points[i][0][0],-points[i][0][1],-1,points[i][1][1]*points[i][0][0],
    #                             points[i][1][1]*points[i][0][1], points[i][1][1]]
    #         amatrix[2*index +1] = [points[i][0][0],points[i][0][1],1,0,0,0,-points[i][1][0]*points[i][0][0],
    #                                -points[i][1][0]*points[i][0][1],-points[i][1][0]]
    #
    #     # print(amatrix)
    #     uvalue, dvalue, vvalue = np.linalg.svd(amatrix)
    #     vvalueT = np.transpose(vvalue)
    #     solution = vvalueT[:,-1]
    #     homography[0] = solution[0:3]/solution[-1]
    #     homography[1] = solution[3:6]/solution[-1]
    #     homography[2] = solution[6:9]/solution[-1]
    #     # homography =np.linalg.pinv(homography)
    #     # homography =homography/homography[2][2]
    #     # print(homography)
    #     return homography

    def perform_ransac(self,tags, samplesize=6, cutoff=3):

        correspondence = self.correspondence[tags[2]]
        # print(correspondence)
        src_xy = np.zeros((len(correspondence), 2))
        dest_xy = np.zeros((len(correspondence), 2))
        src_xy = correspondence[:, 0:2]
        dest_xy = correspondence[:, 2:]
        sourcepoints = []
        destinationpoints = []
        # sx = np.asarray(list(correspondence.keys()))
        # dx = np.asarray(list(correspondence.values()))

        # for key,value in correspondence.items():
        #     sourcepoints.append(key[0])
        #     sourcepoints.append(key[1])
        #     sourcepoints.append(1.0)
        #     destinationpoints.append(value[0])
        #     destinationpoints.append(value[1])
        #     destinationpoints.append(1.0)
        # sourcepoints = np.array(sourcepoints, dtype='float64')
        # sourcepoints = sourcepoints.reshape(-1,3).T
        # destinationpoints = np.array(destinationpoints, dtype='float64')
        # destinationpoints = destinationpoints.reshape(-1,3).T
        count = 0
        listofinliersfinal =[]
        listofoutliersfinal = []
        homographyfinal =np.zeros((3,3))

        for iteration in range(self.ransactrials):
            print(str(iteration) + " of " + str(self.ransactrials))
            print(len(src_xy))
            ip_index = np.random.randint(0, len(src_xy), samplesize)
            src_pts_trial = src_xy[ip_index, :]
            dest_pts_trial = dest_xy[ip_index, :]

            # Calculate Homography by SVD for n selected correspondences
            H = self.LinearLeastSquaresHomography(src_pts_trial, dest_pts_trial)
            # samples =random.sample(list(correspondence.items()),samplesize)
            # print(len(samples))
            # H = self.calculate_lls_homography(src_pts_trial, dest_pts_trial)
            dest_pts_estimate = np.zeros((dest_xy.shape), dtype='int')

            for src_pt in range(len(src_xy)):
                dest_pts_nonNorm = np.matmul(H, ([src_xy[src_pt, 0], src_xy[src_pt, 1], 1]))
                dest_pts_estimate[src_pt, 0] = dest_pts_nonNorm[0] / dest_pts_nonNorm[-1]
                dest_pts_estimate[src_pt, 1] = dest_pts_nonNorm[1] / dest_pts_nonNorm[-1]

            dest_pts_estimate_err = dest_pts_estimate - dest_xy
            dest_pts_estimate_err_sq = np.square(dest_pts_estimate_err)
            dist = np.sqrt(dest_pts_estimate_err_sq[:, 0] + dest_pts_estimate_err_sq[:, 1])
            validpointidx = np.where(dist <= cutoff)
            invalidpointidx = np.where(dist > cutoff)
            innlierlist=[]
            outlierlist =[]
            for i,element in enumerate(dist):
                if element <=cutoff:
                    innlierlist.append([src_xy[i][1],src_xy[i][0],dest_pts_estimate[i][1],dest_pts_estimate[i][0] ])
                else:
                    outlierlist.append([src_xy[i][0], src_xy[i][1], dest_xy[i][0], dest_xy[i][1]])


            Inliers = [1 for val in dist if (val < 3)]
            if len(Inliers) > count:
                count = len(Inliers)
                listofinliersfinal =innlierlist
                listofoutliersfinal =outlierlist
                homographyfinal = H

        self.homographydict[tags[2]]=homographyfinal
        print(len(listofinliersfinal))
        print(len(listofoutliersfinal))
        #self.draw_correspondence_inlier_outlier(listofinliersfinal,listofoutliersfinal)
        self.displayImagewithInterestPointsandOutliers(tags,correspondence,homographyfinal,3)

            # estimatehomography =np.linalg.pinv(estimatehomography)
        #     estimatedpoints = np.matmul(estimatehomography,sourcepoints)
        #     # print(estimatedpoints)
        #     estimatedpoints = estimatedpoints/estimatedpoints[2,:]
        #     squaredifference = (estimatedpoints - destinationpoints)**2
        #     # print(squaredifference)
        #     sumdifference =np.sum(squaredifference, axis=0)
        #     # print(sumdifference)
        #     validpointsidx = np.where(sumdifference <= cutoff**2)
        #     print(validpointsidx)
        #     listofinliersleft = [sx[i] for i in validpointsidx[0]]
        #     if len(listofinliersleft) > count:
        #         count = len(listofinliersleft)
        #         listofinliersfinal = listofinliersleft
        #         homographyfinal = estimatehomography
        # print(listofinliersfinal)

    def displayImagewithInterestPointsandOutliers(self, tags, corners, H, delta):
        # Get shape of the output image
        img1 = self.originalImages[int(tags[0])]
        img2 = self.originalImages[int(tags[1])]
        nrows = max(img1.shape[0], img2.shape[0])
        ncol = img1.shape[1] + img2.shape[1]

        # Initialize combined output image
        out_img = np.zeros((nrows, ncol, 3))

        # Copy Image 1 to left half of the output image
        out_img[:img1.shape[0], :img1.shape[1]] = img1

        # Copy Image 2 to right half of the output image
        out_img[:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

        # Seperate source and destination images XY coordinates
        src_pts = np.zeros((len(corners), 2))
        dest_pts = np.zeros((len(corners), 2))
        src_pts = corners[:, 0:2]
        dest_pts = corners[:, 2:]

        inliers_src_list = []  # list of inliers in source image
        inliers_dest_list = []  # list of inliers in source image
        for src_pt in range(len(src_pts)):
            dest_pt_estimate = np.matmul(H, [src_pts[src_pt, 0], src_pts[src_pt, 1], 1])
            dest_pt_estimate = dest_pt_estimate / dest_pt_estimate[-1]
            diff = dest_pt_estimate[0:2] - dest_pts[src_pt, :]
            err_dist_dest_pt = np.sqrt(np.sum(diff ** 2))
            if err_dist_dest_pt < delta:
                inliers_src_list.append(src_pts[src_pt, :])
                inliers_dest_list.append(dest_pts[src_pt, :])
                cv.circle(out_img, (int(src_pts[src_pt, 0]), int(src_pts[src_pt, 1])), 2, (255, 0, 0), 2)
                cv.circle(out_img, (img1.shape[1] + int(dest_pts[src_pt, 0]), int(dest_pts[src_pt, 1])), 2,
                           (255, 0, 0), 2)
                cv.line(out_img, (int(src_pts[src_pt, 0]), int(src_pts[src_pt, 1])),
                         (img1.shape[1] + int(dest_pts[src_pt, 0]), int(dest_pts[src_pt, 1])), (0, 255, 0))
            else:
                cv.circle(out_img, (int(src_pts[src_pt, 0]), int(src_pts[src_pt, 1])), 2, (0, 0, 255), 2)
                cv.circle(out_img, (img1.shape[1] + int(dest_pts[src_pt, 0]), int(dest_pts[src_pt, 1])), 2,
                           (0, 0, 255), 2)
                cv.line(out_img, (int(src_pts[src_pt, 0]), int(src_pts[src_pt, 1])),
                         (img1.shape[1] + int(dest_pts[src_pt, 0]), int(dest_pts[src_pt, 1])), (0, 0, 255))

        cv.imwrite("results/"+str(tags[2])+"inlieroutlier.jpg", out_img)
        #return out_img, np.array(inliers_src_list), np.array(inliers_dest_list)

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
        # print(self.correspondence[tags[2]])


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
    tester.get_panorama_done()
