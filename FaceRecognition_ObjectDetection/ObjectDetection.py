"""
------------------------------------
------------------------------------
Computer Vision - Purdue University - Homework 11
------------------------------------
Face Recognition & Object Detection
------------------------------------
Author : Arjun Kramadhati Gopi, MS-Computer & Information Technology, Purdue University.
Date: Dec 2, 2020
------------------------------------
[TO RUN CODE]: python3 ObjectDetection.py
------------------------------------
------------------------------------
"""
import os
import pickle
import cv2 as cv
import numpy as np
from tqdm import tqdm
from scipy import spatial
from scipy.linalg import null_space
from scipy import optimize
import matplotlib.pyplot as plt


class ViolaJonesOD:

    def __init__(self, data_path):
        self.image_path = data_path

    def scheduler(self):
        pass


if __name__ == "__main__":
    """
    Code starts here

    """
    tester = ViolaJonesOD(['ECE661_2020_hw11_DB1/test','ECE661_2020_hw11_DB1/train'])
    tester.scheduler()


