"""
Computer Vision - Purdue University - Homework 4

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: September 27, 2020


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

    def __init__(self, image_addresses, scale, kvalue=0):
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
        mvalue = int(np.ceil(4 * self.scale))
        mvalue = mvalue + 1 if (mvalue % 2) > 0 else mvalue
        blankfilter = np.ones((mvalue, mvalue))
        blankfilter[:, :int(mvalue / 2)] = -1
        self.filters["HaarFilterX"] = blankfilter
        blankfilter = np.ones((mvalue, mvalue))
        blankfilter[int(mvalue / 2):, :] = - 1
        self.filters["HaarFilterY"] = blankfilter

    def filter_corner_points(self, rscore, windowsize, queueImage, tag):
        window = int(windowsize / 2)
        for column in range(window, self.grayscaleImages[queueImage].shape[1] - window, 1):
            for row in range(window, self.grayscaleImages[queueImage].shape[0] - window, 1):
                panwindow = rscore[row - window:row + window +1, column - window : column + window +1]
                print(panwindow.shape)
                if rscore[row, column] == np.amax(panwindow):
                    pass
                else:
                    rscore[row, column] = 0

        self.cornerpointdict[tag] = np.asarray(np.where(rscore > 0))
        print(len(np.asarray(np.where(rscore > 0))[0]))

    def draw_corner_points(self, queueImage, tag):

        points = self.cornerpointdict[tag].flatten()
        pointXs = points[:int(len(points)/2)]
        pointYs = points[int(len(points) / 2):]
        image = copy.deepcopy(self.originalImages[queueImage])
        for index in range(len(pointXs)):
            cv.circle(image, (pointYs[index], pointXs[index]), 4, [255, 255, 255], 10)
        return image

    def determine_corners(self, type, queueImage, tag):
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
            tracevalue = sumofdysquared * sumofdxsquared
            if self.kvalue == 0:
                self.kvalue = detvalue / (tracevalue * tracevalue + 0.000001)
                self.kvalue = np.sum(self.kvalue) / (
                            self.grayscaleImages[queueImage].shape[0] * self.grayscaleImages[queueImage].shape[1])
            Rscore = detvalue - (self.kvalue * tracevalue * tracevalue)
            Rscore = np.where(Rscore < 0, 0, Rscore)
            print(Rscore.shape)
            self.filter_corner_points(Rscore, 29, queueImage, tag)

    # def harris_correspondence(self,queueImages, tags):

    def get_sliding_windows(self, windowsize, queueImage,tag, dict, dicttag):

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
        windowsone = copy.deepcopy(self.slidingwindowdict[tags[0]])
        windowstwo = copy.deepcopy(self.slidingwindowdict[tags[1]])
        list = []
        list2 = []
        if style =="SSD":
            for id1 in windowsone:
                list =[]
                list2 =[]
                for id2 in windowstwo:
                    difference = windowsone[id1]-windowstwo[id2]
                    ssd = np.sum(difference*difference)
                    list.append(ssd)
                    list2.append(id2)
                ssd = min(list)
                id = list2[list.index(ssd)]
                windowsone[id1] = (id,ssd)
            self.correspondence[tags[2]] = windowsone
        if style =="NCC":
            for id1 in windowsone:
                list =[]
                list2=[]
                for id2 in windowstwo:
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
                windowsone[id1]=(id,ncc)
            self.correspondence[tags[2]]=windowsone

    def draw_correspondence(self, tags, cutoffvalue, style):
        # ssdvalues = dict(sorted(self.correspondence[tags[1]].items(), key=lambda x: x[1], reverse=True))
        # ssdvalues = dict(filter(lambda x: x[1]>cutoffvalue, ssdvalues.items()))
        # copydict =dict([value,key] for key,value in self.correspondence[tags[0]].items())
        copydict = copy.deepcopy(self.correspondence[tags[0]])
        print(copydict)
        for (key,value) in self.correspondence[tags[0]].items():
            if style == 'greaterthan':
                if value[1]> cutoffvalue:
                    copydict.pop(key)
            elif style == 'lesserthan':
                if value[1]< cutoffvalue:
                    copydict.pop(key)
        resultImage = np.hstack((self.grayscaleImages[0], self.grayscaleImages[1]))
        horizontaloffset = 640
        print(copydict)
        for (key,value) in copydict.items():
            # print((key,value))
            columnvalueone = key[1]
            rowvalueone = key[0]
            columnvaluetwo = value[0][1] + horizontaloffset
            rowvaluetwo = value[0][0]
            cv.line(resultImage, (columnvalueone,rowvalueone), (columnvaluetwo,rowvaluetwo), [255, 255, 255], 1 )
            cv.circle(resultImage,(columnvalueone, rowvalueone), 4, [255, 255, 255], 10)
            cv.circle(resultImage, (columnvaluetwo, rowvaluetwo), 4, [255, 255, 255], 10)
        return resultImage

    def sift_corner_detect(self, queueImage, tag):
        keypoint, descriptor = self.siftobject.detectAndCompute(self.grayscaleImages[queueImage], None)
        self.cornerpointdict[tag] = (keypoint, descriptor)
        img = cv.drawKeypoints(self.grayscaleImages[queueImage], keypoint, self.originalImages[queueImage], flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite(tag + '.jpg', img)

    def sift_correpondence(self, queueImages, tags, method):
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
                tempdict[(int(keypoint1[index].pt[0]),int(keypoint1[index].pt[1]))]=((int(id.pt[0]),int(id.pt[1])),minimumvalue)
            self.correspondence[tags[2]]= tempdict


if __name__ == "__main__":
    tester = FeatureOperator(['hw4_Task1_Images/pair2/1.jpg','hw4_Task1_Images/pair2/2.jpg'], 3.407)
    # tester.build_haar_filter()
    # tester.determine_corners(1, 0, "Harris1")
    # tester.determine_corners(1, 1, "Harris2")
    # image = tester.draw_corner_points(0,"Harris1")
    # cv.imwrite("1.jpg", image)
    # image = tester.draw_corner_points(1, "Harris2")
    # cv.imwrite("2.jpg", image)
    # thread_image_one = threading.Thread(target=tester.get_sliding_windows, args=(21,0,"Harris1",dict(),"Image1HarrisSW",))
    # thread_image_two = threading.Thread(target=tester.get_sliding_windows, args=(21, 1, "Harris2", dict(), "Image2HarrisSW", ))
    # thread_image_one.start()
    # thread_image_two.start()
    # thread_image_one.join()
    # thread_image_two.join()
    # # tester.calculate_correspondence("SSD", ("Image1HarrisSW", "Image2HarrisSW","Image1to2SSD", "Image1to2SSDValues"))
    # tester.calculate_correspondence("NCC", ("Image1HarrisSW", "Image2HarrisSW", "Image1to2NCC", "Image1to2NCCValues"))
    # image = tester.draw_correspondence(("Image1to2NCC", "Image1to2NCCValues"), 0.97, 'greaterthan')
    # cv.imwrite("result.jpg", image)

    tester.sift_corner_detect(0, "Sift1")
    tester.sift_corner_detect(1, "Sift2")
    tester.sift_correpondence((0, 1), ("Sift1","Sift2","Image1to2Eucledian"),'Custom')
    image=tester.draw_correspondence(("Image1to2Eucledian","Image1to2Eucledianvalues"),200, 'greaterthan')
    cv.imwrite("result.jpg", image)