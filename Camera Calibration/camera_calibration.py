"""
Computer Vision - Purdue University - Homework 8

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Oct 19, 2020


[TO RUN CODE]: python3 deeplearnclassifier.py
Output:
    [labels]: Predictions for the input images in the form of a confusion matrix.
"""
import re
import glob
import pickle
import cv2 as cv
import numpy as np
from sklearn import svm
from scipy import signal
from sklearn.model_selection import train_test_split
