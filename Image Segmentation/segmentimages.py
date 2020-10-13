"""
Computer Vision - Purdue University - Homework 6

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Oct 12, 2020


[TO RUN CODE]: python3 segmentimages.py
Output:
    [jpg]: Segmented image which shows the foreground separated from the background.
"""

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import copy


class ImageSegmentation:
    def __init__(self, image_addresses):
        self.image_addresses = image_addresses
        self.originalImages = []
        self.grayscaleImages = []
        self.rgbchannelsdict = {}
        for i in range(len(self.image_addresses)):
            self.originalImages.append(cv.resize(cv.imread(self.image_addresses[i], cv.IMREAD_COLOR), (640, 480)))
            self.grayscaleImages.append(
                cv.resize(cv.imread(self.image_addresses[i], cv.IMREAD_GRAYSCALE), (640, 480)))

    def split_channels(self, inputstyle = 'BGR', gaussianblur = True):
        """
        Splits the RGB image into the three channels based on the way the image is read.
        :param inputstyle: BGR is image is read using cv.imread()
        :param gaussianblur: Yes/No image smoothening
        :return: save the individual channels in dictionary
        """
        for queue in range(len(self.originalImages)):
            if inputstyle == 'BGR':
                r,g,b= self.originalImages[queue][:,:,2], self.originalImages[queue][:,:,1], self.originalImages[queue][:,:,0]
            elif inputstyle =='RGB':
                r, g, b = self.originalImages[queue][:, :, 0], self.originalImages[queue][:, :, 1], self.originalImages[queue][:, :, 2]
            if gaussianblur:
                r,g,b = cv.GaussianBlur(r, (5,5), 0), cv.GaussianBlur(g, (5,5), 0), cv.GaussianBlur(b, (5,5), 0)
            cv.imwrite(str(queue)+'Rchanneloriginal.jpg',r)
            cv.imwrite(str(queue) + 'Gchanneloriginal.jpg', g)
            cv.imwrite(str(queue) + 'Bchanneloriginal.jpg', b)
            self.rgbchannelsdict[queue]={'R': r, 'G': g, 'B': b}

    def filter_masks(self, image):
        """
        Filter the output of the OTSU function using a median blur function.
        :param image: Input image
        :return: returns the filtered image
        """
        filteredimage = cv.medianBlur(image, 13)
        filteredimage = filteredimage > 240
        filteredimage = np.array(filteredimage*255, np.uint8)
        return filteredimage

    def otsu_algorithm(self, image, imagequeue, index, method, iterations):
        """
        Iterative Otsu algorithm implementation.
        :param image: Input image channel
        :param imagequeue: Index value of the image position in the list
        :param index: -
        :param method: RGB/Texture based implementation
        :param iterations: Number of iterations to run OTSU
        :return: returns the OTSU threshold value
        """
        otsucutoff = 0
        otsucutoffinitial = 0
        templist = []
        mask = None
        for iteration in range(iterations):

            start = 0
            # otsucutoffinitial = otsucutoff
            end = 256
            diff = end - start
            channelhistogram = cv.calcHist([np.uint8(image)], [0], mask, [diff], [start, end])
            levels = np.reshape(np.add(range(diff), 1), (diff, 1))
            maxlambda = -1
            otsucutoff = -1
            plt.hist(image.ravel(), diff, [start, end])
            # plt.savefig('histograms/' + str(imagequeue) + str(index) + method)
            plt.show()
            for i in range(len(channelhistogram)):
                m0k = np.sum(channelhistogram[:i]) / np.sum(channelhistogram)
                m1k = np.sum(np.multiply(channelhistogram[:i], levels[:i])) / np.sum(channelhistogram)
                m11k = np.sum(np.multiply(channelhistogram[i:], levels[i:])) / np.sum(channelhistogram)
                omega0 = m0k
                omega1 = 1 - m0k
                mu0 = m1k / omega0
                mu1 = m11k / omega1
                sqauredifference = np.square(mu1 - mu0)
                lambdavalue = omega0 * omega1 * sqauredifference
                if lambdavalue > maxlambda:
                    maxlambda = lambdavalue
                    otsucutoff = i
            mask = np.zeros(image.shape[:2], np.uint8)
            mask[:,:] = image >= otsucutoff
            mask = mask[:,:]*255
            # templist.append(otsucutoff)
            print(otsucutoff)
        # otsucutoff = np.sum(templist)
        # print(otsucutoff)
        return otsucutoff

    def run_otsu_texture(self, imagequeue, iterations, windows):
        """
        Texture based image segmenting implementation. We create three channels based
        on the three window sizes. Each channel consists of the result from convoluting the
        respective window.
        :param imagequeue: Index value of the image position in the list
        :param iterations: Number of iterations to run OTSU
        :param windows: window sizes to extract texture
        :return: draw and save the images with the contours and the foreground representations
        """
        templist = []
        greyimage = self.grayscaleImages[imagequeue]

        for index, window in enumerate(windows):
            textureimgfinal = np.zeros((self.originalImages[imagequeue].shape))
            textureimg = np.zeros((self.grayscaleImages[imagequeue].shape))
            windowsize = np.uint8((window-1)/2)
            for row in range(windowsize, greyimage.shape[0]-windowsize):
                for column in range(windowsize, greyimage.shape[1] - windowsize):
                    slidingwindow = greyimage[row-windowsize:row+windowsize+1, column-windowsize:column+windowsize+1]
                    slidingwindow = slidingwindow - np.mean(slidingwindow)
                    textureimg[row, column] = np.var(slidingwindow)
                    # textureimg[row, column] = np.mean((slidingwindow - np.mean(slidingwindow))**2)
            textureimgfinal[:,:,index] = np.uint8(textureimg/textureimg.max()*255)
            # textureimgfinal[:, :, index] = textureimg*255
            image = textureimgfinal[:, :, index]
            otsucutoff = self.otsu_algorithm(image, imagequeue, index, 'texture', iterations)
            resultimage = np.zeros(self.originalImages[imagequeue].shape)
            print(otsucutoff)
            resultimage[:,:, index] = image <= otsucutoff
            templist.append(resultimage)
            resultimage = resultimage[:,:,index]*255
            cv.imwrite(str(imagequeue)+str(window)+ '.jpg', resultimage)
        combinedimage = np.array(np.logical_and(np.logical_and(templist[0][:, :, 0], templist[1][:, :, 1]),
                                                    templist[2][:, :, 2]) * 255, np.uint8)
        cv.imwrite(str(imagequeue)+'combinedTexturebased.jpg', combinedimage)
        resultimage = self.filter_masks(combinedimage)
        self.draw_foreground_save(resultimage, imagequeue, 'texturemethod')
        self.draw_contour_save(self.extract_contour(resultimage), imagequeue, 'texturemethod')

    def run_otsu_rgb(self, imagequeue, iterations):
        """
        RGB based image segmenting implementation. Using the three split channels, we run
        the OTSU alorithm on each of the channels. We then filter the images based on the
        OTSU cutoff. We combine the three images which will then be used by the contour
        extractor.
        :param imagequeue: Index value of the image position in the list
        :param iterations: Number of iterations to run OTSU
        :return: draw and save the images with the contours and the foreground representations

        """
        templist = []
        for index, channel in enumerate(['R','G','B']):
            image = self.rgbchannelsdict[imagequeue][channel]
            otsucutoff = self.otsu_algorithm(image, imagequeue, index, 'rgb', iterations)
            resultimage = np.zeros(self.originalImages[imagequeue].shape)
            resultimage[:,:, index] = image <= otsucutoff
            templist.append(resultimage)
            resultimage = resultimage[:,:,index]*255
            cv.imwrite(str(imagequeue)+channel + '.jpg', resultimage)
        combinedimage = np.array(np.logical_and(np.logical_and(templist[0][:, :, 0], templist[1][:, :, 1]),
                                                    templist[2][:, :, 2]) * 255, np.uint8)
        cv.imwrite(str(imagequeue)+'combinedRGBbased.jpg', combinedimage)
        resultimage = self.filter_masks(combinedimage)
        self.draw_foreground_save(resultimage, imagequeue, 'rgbmethod')
        self.draw_contour_save(self.extract_contour(resultimage), imagequeue, 'rgbmethod')

    def draw_foreground_save(self, image, imagequeue, method):
        """
        Draw the foreground as black using the contour extracted.
        :param image: Input image
        :param imagequeue: Index value of the image position in the list
        :param method: RGB/Texture based implementation
        :return: Save the image
        """
        r, g, b = self.rgbchannelsdict[imagequeue]['R'], self.rgbchannelsdict[imagequeue]['G'], self.rgbchannelsdict[imagequeue]['B']
        r,g,b = copy.deepcopy(r),copy.deepcopy(g),copy.deepcopy(b)
        truthplot = np.logical_and(np.logical_not(image),1)
        b[truthplot] = 0
        g[truthplot] = 0
        r[truthplot] = 0
        resultimage = cv.merge([b, g, r])
        cv.imwrite(str(imagequeue) +method+ 'foreground.jpg', resultimage)

    def draw_contour_save(self, contours, imagequeue, method):
        """
        Draw the foreground as black using the contour extracted.
        :param contours: Contours we extracted using the extraction algorithm.
        :param imagequeue: Index value of the image position in the list
        :param method: RGB/Texture based implementation
        :return: Save the image
        """
        r,g,b = self.rgbchannelsdict[imagequeue]['R'],self.rgbchannelsdict[imagequeue]['G'],self.rgbchannelsdict[imagequeue]['B']
        r, g, b = copy.deepcopy(r), copy.deepcopy(g), copy.deepcopy(b)
        truthplot = np.logical_and(contours,1)
        b[truthplot] = 0
        g[truthplot] = 255
        r[truthplot] = 0
        resultimage = cv.merge([b,g,r])
        cv.imwrite(str(imagequeue)+method+'contourplot.jpg', resultimage)

    def extract_contour(self, image, style = 1):
        """
        Function to extract the contours from the post-OTSU comined image. For all
        purposes we use the 8-neighbors window size to find the border pixels.
        :param image: Input image of the combined channels
        :param style: 1 for 8 neighbors, 2 for 4 neighbors
        :return: Returns the array of the contours. 255 for border pixels. 0 for others.
        """
        contourplot = np.zeros((image.shape[0],image.shape[1]))
        for row in range(1, image.shape[0]-1):
            for column in range(1, image.shape[1]-1):
                # print(str(column) + " out of " + str(image.shape[0]))
                if image[row,column] == 0:
                    if style == 1:
                        window = image[row-1:row+2, column-1:column+2]
                        if 255 in window:
                            contourplot[row, column] = 255
                    elif style ==2:
                        if(image[row+1, column] == 255 or image[row - 1,column] == 255 or image[row, column+1] == 0 or image[row, column-1] == 0):
                            contourplot[row, column] = 255
        return contourplot


if __name__ == '__main__':

    """
    Code starts here.
    """
    tester = ImageSegmentation(['hw6_images/cat.jpg','hw6_images/pigeon.jpg','hw6_images/Red-Fox_.jpg'])
    tester.split_channels()
    iterations_rgb = [1,2,2]
    iterations_texture = [1,1,1]
    windows=[[5,7,9],[9,11,13],[19,21,23]]
    for i in range(len(iterations_rgb)):
        tester.run_otsu_rgb(i, iterations_rgb[i])
    for i in range(len(iterations_texture)):
        tester.run_otsu_texture(i, iterations_texture[i], windows)
