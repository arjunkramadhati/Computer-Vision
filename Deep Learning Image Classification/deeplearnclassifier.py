"""
Computer Vision - Purdue University - Homework 8

Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Oct 19, 2020


[TO RUN CODE]: python3 deeplearnclassifier.py
Output:
    [labels]: Predictions for the input images
"""

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math
import BitVector
import pickle
import os
from collections import Counter
from scipy import signal
from sklearn.model_selection import train_test_split
import re
from sklearn import svm
