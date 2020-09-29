"""
Computer Vision - Purdue University - Homework 4

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: September 28, 2020


[TO RUN CODE]: python3 cornerdetector.py
Output:
    [jpg]: [Transformed images]
"""

import cv2 as cv
import math
import numpy as np
import time
from scipy import signal as sg
import tqdm
import copy
import threading


class FeatureOperator:

    def __init__(self, image_addresses, scale, kvalue=0.04):
        """
        Initialise the FeatureOperator object. This class can detect corner in an image either by the custom
        built Harris detector or the OpenCV SIFT corner detector. The class can also detect correspondence in
        a given pair of similar pictures either by SSD or NCC or the eucledian distance method.
        :param image_addresses: List of images to work with. In this case it a pair of images.
        :param scale: Scale value used to detecting the corners.
        :param kvalue: K value is the constant used in the equation to calculate the Harris response.
        """
        self.image_addresses = image_addresses
        self.scale = scale
        self.originalImages = []
        self.grayscaleImages = []
        self.imagesizes = []
        self.filters = {}
        self.cornerpointdict = {}
        self.slidingwindowdict = {}
        self.correspondence = {}
        self.kvalue = kvalue
        for i in range(len(self.image_addresses)):
            self.originalImages.append(cv.resize(cv.imread(self.image_addresses[i]),(640,480)))
            self.grayscaleImages.append(cv.resize(cv.cvtColor(cv.imread(self.image_addresses[i]), cv.COLOR_BGR2GRAY),(640,480)))
        self.siftobject = cv.SIFT_create()

    def build_haar_filter(self):
        """
        Builds the two Haar filters to obtain dx and dy values by convolving over the images
        :return: None. Adds the filters to a global dictionary self.filters
        """
        mvalue = int(np.ceil(4 * self.scale))
        mvalue = mvalue + 1 if (mvalue % 2) > 0 else mvalue
        blankfilter = np.ones((mvalue, mvalue))
        blankfilter[:, :int(mvalue / 2)] = -1
        self.filters["HaarFilterX"] = blankfilter
        blankfilter = np.ones((mvalue, mvalue))
        blankfilter[int(mvalue / 2):, :] = - 1
        self.filters["HaarFilterY"] = blankfilter

    def filter_corner_points(self, rscore, windowsize, queueImage, tag):
        """
        From the list of the proposed potential corner points, we apply a type of non-maxima suppression
        to avoid the case of 'overlapping interest points'. That is, we pick the most promisin corner point
        by picking only the one with the highest R-score or Harris response in a given window.
        :param rscore: Harris response as calculated for all the pixels in the image
        :param windowsize: This is currently set to a 29x29 pixel window size.We filter for corner points
        within these windows.
        :param queueImage: Index of the location at which the image under consideration is stored in the list
        :param tag: Values to access and store values by key in the dictionaries
        :return: None. Adds values to a global dictionary self.cornerpointdict
        """

        window = int(windowsize / 2)
        for column in range(window, self.grayscaleImages[queueImage].shape[1] - window, 1):
            for row in range(window, self.grayscaleImages[queueImage].shape[0] - window, 1):
                panwindow = rscore[row - window:row + window +1, column - window : column + window +1]
                #print(panwindow.shape)
                if rscore[row, column] == np.amax(panwindow):
                    pass
                else:
                    rscore[row, column] = 0
        # self.cornerpointdict[tag] = rscore
        print("Here")
        rscoretemp = copy.deepcopy(rscore)
        rscoretemp =rscoretemp.flatten()
        Rcutoffvalue = rscoretemp[np.argsort(rscoretemp)[-100:]][0]
        self.cornerpointdict[tag] = np.asarray(np.where(rscore >= Rcutoffvalue))
        print("Here")
        #print(len(np.asarray(np.where(rscore > 0))[0]))

    def draw_corner_points(self, queueImage, tag):
        """
        We use this function to draw the corner points detected by the custom built Harris corner
        detector.
        :param queueImage: Index of the location at which the image under consideration is stored in the list
        :param tag: Values to access and store values by key in the dictionaries
        :return: Returns the image with the corner points drawn
        """
        points = self.cornerpointdict[tag].flatten()
        pointXs = points[:int(len(points)/2)]
        pointYs = points[int(len(points) / 2):]
        image = copy.deepcopy(self.originalImages[queueImage])
        for index in range(len(pointXs)):
            cv.circle(image, (int(pointYs[index]), int(pointXs[index])), 2, [0, 255, 0], 2)
        return image

    def determine_corners(self, type, queueImage, tag):
        """
        This function is used to calculate the Rscore values or the Harris response values. From these
        values, we determine the final list of corner points after filtering them in the filter_corner_points()
        function.
        :param type: To specify the type of method being deployed to find the corner.
        :param queueImage: Index of the location at which the image under consideration is stored in the list
        :param tag: Values to access and store values by key in the dictionaries
        :return: None. Calls the filter_corner_points() function by giving the Rscore values.
        """
        if type == 1:
            # Harris Corner Method
            dx = sg.convolve2d(self.grayscaleImages[queueImage], self.filters["HaarFilterX"], mode='same')
            dy = sg.convolve2d(self.grayscaleImages[queueImage], self.filters["HaarFilterY"], mode='same')
            dxsquared = dx * dx
            dysquared = dy * dy
            dxy = dx * dy
            windowsize = int(5 * self.scale)
            windowsize = windowsize if (windowsize % 2) > 0 else windowsize + 1
            window = np.ones((windowsize, windowsize))
            sumofdxsquared = sg.convolve2d(dxsquared, window, mode='same')
            sumofdysquared = sg.convolve2d(dysquared, window, mode='same')
            sumofdxdy = sg.convolve2d(dxy, window, mode='same')
            detvalue = (sumofdxsquared * sumofdysquared) - (sumofdxdy * sumofdxdy)
            tracevalue = sumofdysquared + sumofdxsquared
            if self.kvalue == 0:
                self.kvalue = detvalue / (tracevalue * tracevalue + 0.000001)
                self.kvalue = np.sum(self.kvalue) / (
                            self.grayscaleImages[queueImage].shape[0] * self.grayscaleImages[queueImage].shape[1])
            Rscore = detvalue - (self.kvalue * tracevalue * tracevalue)
            Rscore = np.where(Rscore < 0, 0, Rscore)
            print(Rscore.shape)
            self.filter_corner_points(Rscore, 29, queueImage, tag)

    def get_sliding_windows(self, windowsize, queueImage,tag, dict, dicttag):
        """
        This function generates the neighborhood boxes around each corner point in the pair of images.
        These neighborhood boxes or some windowsize (21X21) and they are used to calculate the SSD/NCC values
        to determine correspondence points in the given pair of images.
        :param windowsize: Size of neighborhood boxes
        :param queueImage: Index of the location at which the image under consideration is stored in the list
        :param tag: Values to access and store values by key in the dictionaries
        :param dict: Empty dictionary object for the two simultaneous thread that run this function.
        :param dicttag: Values to access and store values by key in the dictionaries
        :return: None. Stores the windows in the global dictionary self.slidingwindowdict
        """
        points = self.cornerpointdict[tag].flatten()
        pointXs = points[:int(len(points)/2)]
        pointYs = points[int(len(points) / 2):]
        for index in range(len(pointXs)):
            row = pointXs[index]
            column = pointYs[index]
            array = self.grayscaleImages[queueImage][row:row+windowsize, column:column+windowsize]
            if array.shape == (29,29):
                dict[(row, column)] = array
            else:
                resultarray = np.zeros((29,29))
                resultarray[:array.shape[0],:array.shape[1]] = array
                dict[(row, column)] = resultarray

        self.slidingwindowdict[dicttag] = dict

    def calculate_correspondence(self, style, tags):
        """
        This function calculates the correspondence between the two images in the pair.
        The methods employed in this function are 1) SSD and 2) NCC. We use either one to
        calculte the correspondence.
        :param style: Either SSD or NCC method to calculate correspondence.
        :param tags: Values to access and store values by key in the dictionaries
        :return: None. Stores the correpondece/the matching pairs of points in the a global dictionary
        self.correspondence
        """
        windowsone = copy.deepcopy(self.slidingwindowdict[tags[0]])
        windowstwo = copy.deepcopy(self.slidingwindowdict[tags[1]])
        list = []
        list2 = []
        list3=[]
        if style =="SSD":
            list3=[]
            for id1 in windowsone:
                list =[]
                list2 =[]
                for id2 in windowstwo:
                    if id2 in list3:
                        pass
                    else:
                        difference = windowsone[id1]-windowstwo[id2]
                        ssd = np.sum(difference*difference)
                        list.append(ssd)
                        list2.append(id2)
                ssd = min(list)
                id = list2[list.index(ssd)]
                list3.append(id)
                windowsone[id1] = (id,ssd)
            self.correspondence[tags[2]] = windowsone
        if style =="NCC":
            list3=[]
            for id1 in windowsone:
                list =[]
                list2=[]
                for id2 in windowstwo:
                    if id2 in list3:
                        pass
                    else:
                        meanvalueone = np.mean(windowsone[id1])
                        meanvaluetwo = np.mean(windowstwo[id2])
                        diffone = windowsone[id1] -meanvalueone
                        difftwo = windowstwo[id2] - meanvaluetwo
                        ssd = np.sum(diffone*difftwo)
                        ssdone = np.sum(diffone*diffone)
                        ssdtwo = np.sum(difftwo*difftwo)
                        list.append(ssd/(np.sqrt(ssdone*ssdtwo)+0.000001))
                        list2.append(id2)
                ncc = max(list)
                id = list2[list.index(ncc)]
                list3.append(id)
                windowsone[id1]=(id,ncc)
            self.correspondence[tags[2]]=windowsone

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
            cv.imwrite("Sift_Correspondence.jpg", result)
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



if __name__ == "__main__":
    """
    Code starts here. After each run, change images to get results for a new set of image pair. 
    """

    tester = FeatureOperator(['hw4_Task1_Images/pair5/1.jpg','hw4_Task1_Images/pair5/2.jpg'], 0.7)
    tester.build_haar_filter()
    tester.determine_corners(1, 0, "Harris1")
    tester.determine_corners(1, 1, "Harris2")
    image = tester.draw_corner_points(0,"Harris1")
    cv.imwrite("1.jpg", image)
    image = tester.draw_corner_points(1, "Harris2")
    cv.imwrite("2.jpg", image)
    thread_image_one = threading.Thread(target=tester.get_sliding_windows, args=(21,0,"Harris1",dict(),"Image1HarrisSW",))
    thread_image_two = threading.Thread(target=tester.get_sliding_windows, args=(21, 1, "Harris2", dict(), "Image2HarrisSW", ))
    thread_image_one.start()
    thread_image_two.start()
    thread_image_one.join()
    thread_image_two.join()
    # tester.calculate_correspondence("SSD", ("Image1HarrisSW", "Image2HarrisSW","Image1to2SSD", "Image1to2SSDValues"))
    tester.calculate_correspondence("SSD", ("Image1HarrisSW", "Image2HarrisSW", "Image1to2NCC", "Image1to2NCCValues"))
    image = tester.draw_correspondence(("Image1to2NCC", "Image1to2NCCValues"), 8000000, 'greaterthan')
    cv.imwrite("result.jpg", image)

    tester.sift_corner_detect(0, "Sift1")
    tester.sift_corner_detect(1, "Sift2")
    tester.sift_correpondence((0, 1), ("Sift1","Sift2","Image1to2Eucledian"),'Custom')
    image=tester.draw_correspondence(("Image1to2Eucledian","Image1to2Eucledianvalues"),80, 'greaterthan')
    cv.imwrite("result.jpg", image)
