"""
Computer Vision - Purdue University - Homework 3
    
Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: September 21, 2020


[TO RUN CODE]: python3 removeDistortion.py
The code displays the pictures. The user will have to select the ROI points manually in the PQRS fashion.
P ------- Q
|         |
|         |
|         |
R ------- S

Output:
    [jpg]: [Transformed images]
"""

import cv2 as cv
import math
import numpy as np
import time


class removeDistortion:

    def __init__(self,image_addresses):
        
        
        self.image_addresses=image_addresses
        self.image_one = cv.imread(image_addresses[0])
        self.image_one = cv.resize(self.image_one,(int(self.image_one.shape[1]*0.5),int(self.image_one.shape[0]*0.5)))
        self.image_two = cv.imread(image_addresses[1])
        # self.image_two = cv.resize(self.image_two,(int(self.image_two.shape[1]*0.3),int(self.image_two.shape[0]*0.3)))
        self.image_three = cv.imread(image_addresses[2])
        self.image_three = cv.resize(self.image_three,(int(self.image_three.shape[1]*0.2),int(self.image_three.shape[0]*0.2)))
        self.images = [self.image_one,self.image_two,self.image_three]
        self.image_sizes = [(self.image_one.shape[0],self.image_one.shape[1]), (self.image_two.shape[0],self.image_two.shape[1]),(self.image_three.shape[0],self.image_three.shape[1])]
        self.image_sizes_corner_points_HC= []
        self.roiRealWorld = [[(0.0,0.0,1.0),(75.0,0.0,1.0),(0.0,85.0,1.0),(75.0,85.0,1.0)],[(0.0,0.0,1.0),(84.0,0.0,1.0),(0.0,74.0,1.0),(84.0,74.0,1.0)],[(0.0,0.0,1.0),(55.0,0.0,1.0),(0.0,36.0,1.0),(55.0,36.0,1.0)],[(0.0,0.0,1.0),(69.0,0.0,1.0),(0.0,31.0,1.0),(69.0,31.0,1.0)]]
        self.roiCoordinates = []
        self.roiList = []
        self.homographies=[]
        self.resultImg = []
        self.xmin = 0
        self.ymin =0
        self.createImageCornerPointRepresentations()


    def createImageCornerPointRepresentations(self):
        """
        [summary] This function creates HC representations of the corner points of the given original input images.
        """
        templist = []
        for size in self.image_sizes:
            templist.append(np.asarray([0.0,0.0,1.0]))
            templist.append(np.asarray([float(size[1])-1.0,0.0,1.0]))
            templist.append(np.asarray([0.0,float(size[0])-1.0,1.0]))
            templist.append(np.asarray([float(size[1])-1.0,float(size[0])-1.0,1.0]))
            self.image_sizes_corner_points_HC.append(templist)
            templist = []


    def append_points(self,event,x,y,flags,param):
        """
        [This function is called every time the mouse left button is clicked - It records the (x,y) coordinates of the click location]
        
        """
        if event == cv.EVENT_LBUTTONDOWN:
            self.roiCoordinates.append((float(x),float(y),1.0))
            

    
    def getROIFromUser(self):
        """
        [This function is responsible for taking the regions of interests from the user for all the 4 pictures in order]
        
        """
        self.roiList=[]
        cv.namedWindow('Select ROI')
        
        cv.setMouseCallback('Select ROI',self.append_points)
        for i in range(3):
            while(True):
                cv.imshow('Select ROI',self.images[i])
                k = cv.waitKey(1) & 0xFF
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
             
            self.roiList.append(self.roiCoordinates)
            
            self.roiCoordinates = []
        
    
    def weightedPixelValue(self,rangecoordinates,objectQueue):
        """
        [This function calculates the weighted pixel value at the given coordinate in the target image]

        Args:
            rangecoordinates ([list]): [This is the coordinate of the pixel in the target image]
            objectQueue ([int]): [This is the index number of the list which has the coordinates of the roI for the Object picture]

        Returns:
            [list]: [Weighted pixel value - RGB value]
        """

        pointOne = (int(np.floor(rangecoordinates[1])),int(np.floor(rangecoordinates[0])))
        pointTwo = (int(np.floor(rangecoordinates[1])),int(np.ceil(rangecoordinates[0])))
        pointThree = (int(np.ceil(rangecoordinates[1])),int(np.ceil(rangecoordinates[0])))
        pointFour = (int(np.ceil(rangecoordinates[1])),int(np.floor(rangecoordinates[0])))

        pixelValueAtOne = self.images[objectQueue][pointOne[0]][pointOne[1]]
        pixelValueAtTwo = self.images[objectQueue][pointTwo[0]][pointTwo[1]]
        pixelValueAtThree = self.images[objectQueue][pointThree[0]][pointThree[1]]
        pixelValueAtFour = self.images[objectQueue][pointFour[0]][pointFour[1]]

        weightAtOne = 1/np.linalg.norm(pixelValueAtOne-rangecoordinates)
        weightAtTwo = 1/np.linalg.norm(pixelValueAtTwo-rangecoordinates)
        weightAtThree = 1/np.linalg.norm(pixelValueAtThree-rangecoordinates)
        weightAtFour = 1/np.linalg.norm(pixelValueAtFour-rangecoordinates)
        
        return ((weightAtOne*pixelValueAtOne) + (weightAtTwo*pixelValueAtTwo) + (weightAtThree*pixelValueAtThree) + (weightAtFour*pixelValueAtFour))/(weightAtFour+weightAtThree+weightAtTwo+weightAtOne)


    def createBlankImageArray(self,queueHomography,queueImage):
        """[summary]
        This function is called to create the blank image. The blank image is formed of an array - np.zeros. The size of the blank image is calculated
        based on the homography matrix which is being used. The original corner points are used to calculate the new corner points in the new image. 

        Args:
            queueHomography ([int]): [Index of the homography matrix being used to calculate the new image size]
            queueImage ([int]): [Index of the image in the list being used]

        Returns:
            [numpy array]: [np.zeros of the size equal to the new image size]
            [int]: [Returns the xmin value of the new image - The least x value amongst the four transformed corner points]
            [int]: [Returns the ymin value of the new image - The least y value amongst the four transformed corner points]
        """
        
        templist = []
        templistX=[]
        templistY=[]
        #print(self.image_sizes_corner_points_HC[queueImage])
        #print(self.homographies[queueHomography][0])
        for i in range(4):
            templist.append(np.dot(self.homographies[queueHomography],self.image_sizes_corner_points_HC[queueImage][i]))
        #print(templist)

        for i,element in enumerate(templist):
            templist[i] = element/element[2]
        for element in templist:
            templistX.append(element[0])
            templistY.append(element[1])

        breadth = int(math.ceil(max(templistX))) - int(math.floor(min(templistX)))
        height = int(math.ceil(max(templistY))) - int(math.floor(min(templistY)))

        return np.zeros((height,breadth,3)),int(math.floor(min(templistX))),int(math.floor(min(templistY)))



    def createImage(self,queueHomography,queueImage):
        """[summary]
        This function is the function which creates the final result image. This function has the traditional but slow nested for loop approach to build the image.
        It begins by first getting the blank image of the size of the new image from the createBlankImageArray function above.

        Args:
            queueHomography ([int]): [Index of the homography matrix being used to calculate the new image size]
            queueImage ([int]): [Index of the image in the list being used]

        Returns:
            [numpy ndarray]: [Returns the final resultant image in numpy.ndarray form.]
        """
        print("Processing...")
        resultImg,xmin,ymin = self.createBlankImageArray(queueHomography,queueImage)

        for column in range(0,resultImg.shape[0]):
            for row in range(0,resultImg.shape[1]):
                print("processing" + str(column) + " out of "+ str(resultImg.shape[0]))
                rangecoordinates = np.dot(self.homographies[queueHomography+1],(float(row+xmin),float(column+ymin),1.0))
                rangecoordinates = rangecoordinates/rangecoordinates[2]

                if ((rangecoordinates[0]>0) and (rangecoordinates[0]<self.image_sizes[queueImage][1]-1)) and ((rangecoordinates[1]>0) and (rangecoordinates[1]<self.image_sizes[queueImage][0]-1)):
                    resultImg[column][row] = self.weightedPixelValue(rangecoordinates,queueImage)
                else:
                    resultImg[column][row] = [0,0,0]
        
        return resultImg

    def createImageVectorised(self,queueHomography,queueImage):
        """[summary] 
        ----------- Attempt #1 ---------------
        
        Vectorised numpy operation

        --------------------------------------
        
        This function is the function which creates the final result image. This was the first attempt towards writing a fully vectorised numpy pythonic operation. 
        Here, I first arrange the coordinates of each pixel in a vertical stack (Line 205 - 207). Then I add xmin and y min vallues to each of the X values and Y values. 
        Then I add a third row of just ones to make them into individual 3X1 vectors. Using these stacked vectors of individual pixel coordinates, I perform a vector 
        multiplication with the homograhy matrix H. I do this using the '@' operator. The resulting matrix has the corresponding pixel coordinates of the source image. 
        I extract the pixel values of each of these coordinates using a nested for loop. Basically, I was able to avoid the matrix multiplication being written inside the
        nexted for loop. I was able to get stable outputs much quicker - 40% faster. 

        Args:
            queueHomography ([int]): [Index of the homography matrix being used to calculate the new image size]
            queueImage ([int]): [Index of the image in the list being used]

        Returns:
            [numpy ndarray]: [Returns the final resultant image in numpy.ndarray form.]
        """

        print("processing...")
        resultImg,xmin,ymin = self.createBlankImageArray(queueHomography,queueImage)
        column,row = np.mgrid[0:resultImg.shape[0],0:resultImg.shape[1]]
        vector = np.vstack((column.ravel(),row.ravel()))
        row = vector[1] + xmin
        column = vector[0] +ymin
        ones = np.ones(len(row))
        vector = np.array([column,row,ones])
        s=time.time()
        resultvector = self.homographies[queueHomography+1]@vector
        e=time.time()
        print("timetake",e-s)
        resultvector = resultvector/resultvector[2]
        # resultvector = resultvector[:2,:]
        for column in range(0,resultImg.shape[0]):
            for row in range(0,resultImg.shape[1]):
                print("processing" + str(column) + " out of "+ str(resultImg.shape[0]))

                rangecoordinates=np.array([resultvector[1][(column*resultImg.shape[1])+row],resultvector[0][(column*resultImg.shape[1])+row],resultvector[2][(column*resultImg.shape[1])+row]])

                if ((rangecoordinates[0]>0) and (rangecoordinates[0]<self.image_sizes[queueImage][1]-1)) and ((rangecoordinates[1]>0) and (rangecoordinates[1]<self.image_sizes[queueImage][0]-1)):
                    resultImg[column][row] = self.weightedPixelValue(rangecoordinates,queueImage)
                else:
                    resultImg[column][row] = [255.0,255.0,255.0]
        
        return resultImg




    def buildImage(self,queueHomography,queueImage,row,column):
        """[summary] 
        ----------- Attempt #2 ---------------
        
        Vectorised numpy operation

        --------------------------------------
        
        This function is the function which creates the final result image. This was the second attempt towards writing a fully vectorised numpy pythonic operation. 
        This function is pretty much the same as the createImage function. The ket difference here is that this function does not have the nester for loop. 
        Instead, I vectorise this entire function using the numpy vectorise operation. Using this entire function as a vector, I was able to successfully vectorise the
        whole image building process. 

        Args:
            queueHomography ([int]): [Index of the homography matrix being used to calculate the new image size]
            queueImage ([int]): [Index of the image in the list being used]
            row ([int]) : [Row value of the pixel being considered]
            column ([int]) : [Column value of the pixel being considered]

        Returns:
            Does not return any value. It just updates the global image variable (self.resultImg).
        """
        

        rangecoordinates = np.matmul(self.homographies[queueHomography+1],(float(row+self.xmin),float(column+self.ymin),1.0))
        rangecoordinates = rangecoordinates/rangecoordinates/[2]
        if ((rangecoordinates[0]>0) and (rangecoordinates[0]<self.image_sizes[queueImage][1]-1)) and ((rangecoordinates[1]>0) and (rangecoordinates[1]<self.image_sizes[queueImage][0]-1)):
            pointOne = (int(np.floor(rangecoordinates[1])),int(np.floor(rangecoordinates[0])))
            pointTwo = (int(np.floor(rangecoordinates[1])),int(np.ceil(rangecoordinates[0])))
            pointThree = (int(np.ceil(rangecoordinates[1])),int(np.ceil(rangecoordinates[0])))
            pointFour = (int(np.ceil(rangecoordinates[1])),int(np.floor(rangecoordinates[0])))

            pixelValueAtOne = self.images[queueImage][pointOne[0]][pointOne[1]]
            pixelValueAtTwo = self.images[queueImage][pointTwo[0]][pointTwo[1]]
            pixelValueAtThree = self.images[queueImage][pointThree[0]][pointThree[1]]
            pixelValueAtFour = self.images[queueImage][pointFour[0]][pointFour[1]]

            weightAtOne = 1/np.linalg.norm(pixelValueAtOne-rangecoordinates)
            weightAtTwo = 1/np.linalg.norm(pixelValueAtTwo-rangecoordinates)
            weightAtThree = 1/np.linalg.norm(pixelValueAtThree-rangecoordinates)
            weightAtFour = 1/np.linalg.norm(pixelValueAtFour-rangecoordinates)
            
            self.resultImg[column][row] = ((weightAtOne*pixelValueAtOne) + (weightAtTwo*pixelValueAtTwo) + (weightAtThree*pixelValueAtThree) + (weightAtFour*pixelValueAtFour))/(weightAtFour+weightAtThree+weightAtTwo+weightAtOne)
        else:
            
            self.resultImg[column][row] = [255.0,255.0,255.0]

    def vectoriseOperations(self,queueHomography,queueImage):
        """[summary] 
        ----------- Attempt #2 Continued ---------------
        
        Vectorised numpy operation

        --------------------------------------
        
        This function is the extension of the above function - buildImage. This is the function which vectorises the entire buildImage function. 
        In this function, I stack a list which contains all the pixel coordinates in the blank image. I feed this entire list to the vectorised function.
        This was a successful vectorisation operation however the RAM utilization peaked to a hundred percent. The laptop froze and I could not run this further. 

        Args:
            queueHomography ([int]): [Index of the homography matrix being used to calculate the new image size]
            queueImage ([int]): [Index of the image in the list being used]

        Returns:
            [numpy ndarray]: [Returns the final resultant image in numpy.ndarray form.]
        """
        self.resultImg,self.xmin,self.ymin = self.createBlankImageArray(queueHomography,queueImage)
        length = self.resultImg.shape[0]*self.resultImg.shape[1]
        queueHomography = [queueHomography]*length
        queueImage = [queueImage]*length
        vectoriseOperation = np.vectorize(self.buildImage)
        row,column = np.mgrid[0:self.resultImg.shape[1],0:self.resultImg.shape[0]]
        point = np.vstack((row.ravel(),column.ravel()))
        row = point[0]
        column = point[1]
        #print(point)
        print("processing...")
        vectoriseOperation(queueHomography,queueImage,row,column)
        return self.resultImg






    def objectMatrixFunction(self,queue):
        """
        [We construct the B Matrix with dimension 8X1]

        Args:
            queue ([int]): [This is the index number of the list which has the coordinates of the roI for the object picture]
        """
        self.objectMatrix = np.zeros((8,1))
        
        for i in range(len(self.roiRealWorld[queue])):
            self.objectMatrix[(2*i)][0] = self.roiRealWorld[queue][i][0]
            self.objectMatrix[(2*i)+1][0] = self.roiRealWorld[queue][i][1]

    def parameterMatrixFunction(self,queue,objectQueue):
        """
        [We construct the A Matrix with dimension 8X8 and then we calculate the inverse of A matrix needed for the homography calculation]

        Args:
            queue ([int]): [This is the index number of the list which has the coordinates of the roI for the destination picture]
            objectQueue ([int]): [This is the index number of the list which has the coordinates of the roI for the Object picture]
        """
        self.parameterMatrix=np.zeros((8,8))
        
        for i in range(4):
            self.parameterMatrix[2*i][0] = self.roiList[queue][i][0]
            self.parameterMatrix[2*i][1] = self.roiList[queue][i][1]
            self.parameterMatrix[2*i][2] = 1.0
            self.parameterMatrix[2*i][3] = 0.0
            self.parameterMatrix[2*i][4] = 0.0
            self.parameterMatrix[2*i][5] = 0.0
            self.parameterMatrix[2*i][6] = (-1)*(self.roiList[queue][i][0])*(self.roiRealWorld[objectQueue][i][0])
            self.parameterMatrix[2*i][7] = (-1)*(self.roiList[queue][i][1])*(self.roiRealWorld[objectQueue][i][0])
            self.parameterMatrix[(2*i) + 1][0] = 0.0
            self.parameterMatrix[(2*i) + 1][1] = 0.0
            self.parameterMatrix[(2*i) + 1][2] = 0.0
            self.parameterMatrix[(2*i) + 1][3] = self.roiList[queue][i][0]
            self.parameterMatrix[(2*i) + 1][4] = self.roiList[queue][i][1]
            self.parameterMatrix[(2*i) + 1][5] = 1.0
            self.parameterMatrix[(2*i) + 1][6] = (-1)*(self.roiList[queue][i][0])*(self.roiRealWorld[objectQueue][i][1])
            self.parameterMatrix[(2*i) + 1][7] = (-1)*(self.roiList[queue][i][1])*(self.roiRealWorld[objectQueue][i][1])

        self.parameterMatrixI = np.linalg.pinv(self.parameterMatrix)

    def calculateHomography(self):
        """
        [We calculate the homography matrix here. Once we have the values of the matrix, we rearrange them into a 3X3 matrix.]
        
        """
        homographyI = np.matmul(self.parameterMatrixI,self.objectMatrix)
        homography = np.zeros((3,3))

        homography[0][0]= homographyI[0]
        homography[0][1]= homographyI[1]
        homography[0][2]= homographyI[2]
        homography[1][0]= homographyI[3]
        homography[1][1]= homographyI[4]
        homography[1][2]= homographyI[5]
        homography[2][0]= homographyI[6]
        homography[2][1]= homographyI[7]
        homography[2][2]= 1.0
        self.homographies.append(homography)
        homography = np.linalg.pinv(homography)
        homography = homography/homography[2][2]
        self.homographies.append(homography)



    def projectiveDistortionHomography(self,queueImage):
        """[summary]
        Calculate the homography matrix to eliminate projective distortion

        Args:
            queueImage ([int]): [Index of the image in the list being used]

        Calculates the Homography matrix and appends it to the global homography list.
        """

        vanishingPointOne = np.cross(np.cross(self.roiList[queueImage][0],self.roiList[queueImage][1]),np.cross(self.roiList[queueImage][2],self.roiList[queueImage][3]))
        vanishingPointTwo = np.cross(np.cross(self.roiList[queueImage][0],self.roiList[queueImage][2]),np.cross(self.roiList[queueImage][1],self.roiList[queueImage][3]))

        vanishingLine = np.cross((vanishingPointOne/vanishingPointOne[2]),(vanishingPointTwo/vanishingPointTwo[2]))

        projectiveDHomography = np.zeros((3,3))
        projectiveDHomography[2] = vanishingLine/vanishingLine[2]
        projectiveDHomography[0][0] = 1
        projectiveDHomography[1][1] = 1
        self.homographies.append(projectiveDHomography)
        inverseH = np.linalg.pinv(projectiveDHomography)
        self.homographies.append(inverseH/inverseH[2][2])
        


    def affineDistortionHomography(self,queueImage):
        """[summary]
        Calculate the homography matrix to eliminate affine distortion

        Args:
            queueImage ([int]): [Index of the image in the list being used]

        Calculates the Homography matrix and appends it to the global homography list.
        """
        templist = []
        temppoints = []

        for i in range(4):
            tempvalue = np.dot(self.homographies[0],self.roiList[queueImage][i])
            tempvalue = tempvalue/tempvalue[2]
            temppoints.append(tempvalue)
        
        print(temppoints)
        ortholinePairOne = np.cross(temppoints[0],temppoints[1])
        ortholinePairTwo = np.cross(temppoints[0],temppoints[2])
        ortholinePairThree = np.cross(temppoints[0],temppoints[3])
        ortholinePairFour = np.cross(temppoints[1],temppoints[2])
        templist.append(ortholinePairOne)
        templist.append(ortholinePairTwo)
        templist.append(ortholinePairThree)
        templist.append(ortholinePairFour)

        for i,element in enumerate(templist):
            #print(element)
            #print(element[2])
            templist[i] = element/element[2]
        
        matrixAT = []
        matrixAT.append([templist[0][0]*templist[1][0],templist[0][0]*templist[1][1]+templist[0][1]*templist[1][0]])
        matrixAT.append([templist[2][0]*templist[3][0],templist[2][0]*templist[3][1]+templist[2][1]*templist[3][0]])
        matrixAT = np.asarray(matrixAT)
        matrixAT = np.linalg.pinv(matrixAT)
        matrixA = []
        matrixA.append([-templist[0][1]*templist[1][1]])
        matrixA.append([-templist[2][1]*templist[3][1]])
        matrixA = np.asarray(matrixA)

        matrixS = np.dot(matrixAT,matrixA)
        matrixSRearranged = np.zeros((2,2))

        matrixSRearranged[0][0] = matrixS[0]
        matrixSRearranged[0][1] = matrixS[1]
        matrixSRearranged[1][0] = matrixS[1]
        matrixSRearranged[1][1] = 1

        v,lambdamatrix,q = np.linalg.svd(matrixSRearranged)

        lambdavalue = np.sqrt(np.diag(lambdamatrix))
        Hmatrix = np.dot(np.dot(v,lambdavalue),v.transpose())

        affineHomography=np.zeros((3,3))
        affineHomography[0][0] = Hmatrix[0][0]
        affineHomography[0][1] = Hmatrix[0][1]
        affineHomography[1][0] = Hmatrix[1][0]
        affineHomography[1][1] = Hmatrix[1][1]
        affineHomography[2][2] = 1

        
        inverseH = np.linalg.pinv(affineHomography)
        inverseH = np.dot(inverseH,self.homographies[0])
        self.homographies.append(inverseH)
        inverseH = np.linalg.pinv(inverseH)
        self.homographies.append(inverseH/inverseH[2][2])

    def oneStepDistortionHomography(self,queueImage):
        """[summary]
        Calculate the homography matrix to eliminate both projective and affine distortion

        Args:
            queueImage ([int]): [Index of the image in the list being used]

        Calculates the Homography matrix and appends it to the global homography list.
        """
        matrixA=[]
        matrixAT = []
        templist=[]
        templist.append(np.cross(self.roiList[queueImage][0],self.roiList[queueImage][1]))
        templist.append(np.cross(self.roiList[queueImage][1],self.roiList[queueImage][3]))
        templist.append(np.cross(self.roiList[queueImage][1],self.roiList[queueImage][3]))
        templist.append(np.cross(self.roiList[queueImage][3],self.roiList[queueImage][2]))
        templist.append(np.cross(self.roiList[queueImage][3],self.roiList[queueImage][2]))
        templist.append(np.cross(self.roiList[queueImage][2],self.roiList[queueImage][0]))
        templist.append(np.cross(self.roiList[queueImage][2],self.roiList[queueImage][0]))
        templist.append(np.cross(self.roiList[queueImage][0],self.roiList[queueImage][1]))
        templist.append(np.cross(self.roiList[queueImage][0],self.roiList[queueImage][3]))
        templist.append(np.cross(self.roiList[queueImage][1],self.roiList[queueImage][2]))

        for i,element in enumerate(templist):
            templist[i] = element/element[2]
        
        for i in range(0,10,2):
            matrixAT.append([templist[i][0]*templist[i+1][0],(templist[i][0]*templist[i+1][1]+templist[i][1]*templist[i+1][0])/2,templist[i][1]*templist[i+1][1],(templist[i][0]*templist[i+1][2]+templist[i][2]*templist[i+1][0])/2,(templist[i][1]*templist[i+1][2]+templist[i][2]*templist[i+1][1])/2])
            matrixA.append([-templist[i][2]*templist[i+1][2]])
        
        matrixAT = np.asarray(matrixAT)
        matrixA = np.asarray(matrixA)
        matrixS = np.dot(np.linalg.pinv(matrixAT),matrixA)
        matrixS = matrixS/np.max(matrixS)

        matrixSRearranged = np.zeros((2,2))
        matrixSRearranged[0][0] = matrixS[0]
        matrixSRearranged[0][1] = matrixS[1] * 0.5
        matrixSRearranged[1][0] = matrixS[1] * 0.5
        matrixSRearranged[1][1] = matrixS[2]
        matrixST = np.array([matrixS[3]*0.5,matrixS[4]*0.5])
        v,lambdamatrix,q = np.linalg.svd(matrixSRearranged)
        lambdavalue = np.sqrt(np.diag(lambdamatrix))
        Hmatrix = np.dot(np.dot(v,lambdavalue),v.transpose())
        Vmatrix = np.dot(np.linalg.pinv(Hmatrix),matrixST)

        onestepHomography =np.zeros((3,3))
        onestepHomography[0][0] = Hmatrix[0][0]
        onestepHomography[0][1] = Hmatrix[0][1]
        onestepHomography[1][0] = Hmatrix[1][0]
        onestepHomography[1][1] = Hmatrix[1][1]
        onestepHomography[2][0] = Vmatrix[0]
        onestepHomography[2][1] = Vmatrix[1]
        onestepHomography[2][2]=1

        inverseH = np.linalg.pinv(onestepHomography)
        self.homographies.append(inverseH)
        inverseH = np.linalg.pinv(inverseH)
        self.homographies.append(inverseH/inverseH[2][2])
        




if __name__ == "__main__":

    tester = removeDistortion(['hw3_Task1_Images/Images/1.jpg','hw3_Task1_Images/Images/2.jpg','hw3_Task1_Images/Images/3.jpg'])
    tester.getROIFromUser()
    for i in range(0,3):
        tester.objectMatrixFunction(i)
        tester.parameterMatrixFunction(i,i)
        tester.calculateHomography()
        resultImg = tester.createImage(0,i)
        cv.imwrite("ptp" +str(i)+".jpg",resultImg)
        
    tester.getROIFromUser()

    for i in range(0,3):
        tester.projectiveDistortionHomography(i)
        resultImg = tester.createImage(0,i)
        # resultImg = tester.createImageVectorised(0,0)
        cv.imwrite('1' +str(i)+'.jpg',resultImg)
        tester.affineDistortionHomography(i)
        resultImg = tester.createImage(2,i)
        cv.imwrite('2' +str(i)+'.jpg',resultImg)
        tester.oneStepDistortionHomography(i)
        resultImg = tester.createImage(4,i)
        cv.imwrite('3' +str(i)+'.jpg',resultImg)

    ######Custom Input Images########

    tester = removeDistortion(['hw3_Task1_Images/Images/sn.jpg','hw3_Task1_Images/Images/laptop.jpg'])
    tester.getROIFromUser()
    for i in range(0,2):
        tester.objectMatrixFunction(i)
        tester.parameterMatrixFunction(i,i)
        tester.calculateHomography()
        resultImg = tester.createImage(0,i)
        cv.imwrite("ptp" +str(i)+".jpg",resultImg)
        
    tester.getROIFromUser()

    for i in range(0,2):
        tester.projectiveDistortionHomography(i)
        resultImg = tester.createImage(0,i)
        # resultImg = tester.createImageVectorised(0,0)
        cv.imwrite('1' +str(i)+'.jpg',resultImg)
        tester.affineDistortionHomography(i)
        resultImg = tester.createImage(2,i)
        cv.imwrite('2' +str(i)+'.jpg',resultImg)
        tester.oneStepDistortionHomography(i)
        resultImg = tester.createImage(4,i)
        cv.imwrite('3' +str(i)+'.jpg',resultImg)
    

        