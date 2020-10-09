"""
Computer Vision - Purdue University - Homework 6

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Oct 5, 2020


[TO RUN CODE]: python3 segmentimages.py
Output:
    [jpg]: Segmented image which shows the foreground separated from the background.
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

class ImageSegmentation:
    def __init__(self, image_addresses):
        self.image_addresses = image_addresses
        self.originalImages = []
        self.grayscaleImages = []
        self.filters = {}
        self.cornerpointdict = {}
        self.slidingwindowdict = {}
        self.correspondence = {}
        self.homographydict = {}
        for i in range(len(self.image_addresses)):
            self.originalImages.append(cv.resize(cv.imread(self.image_addresses[i]), (640, 480)))
            self.grayscaleImages.append(
                cv.resize(cv.cvtColor(cv.imread(self.image_addresses[i]), cv.COLOR_BGR2GRAY), (640, 480)))