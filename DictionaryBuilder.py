#!/usr/bin/python

# Function used to build dictionary of code words

# Algorithm
# 1. For given subset of images, get the SURF descriptors for each of the image
# 2. Plot these descriptors in high dimensional space
# 3. Run k-means to computer 50 clusters and return these clusters as dictionary

# TODO:
# Have better logic of computing the dictionary

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import Utils

def findSurfDescriptor(filename):
    img = cv2.imread(filename)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.SURF(hessianThreshold=100, extended=False)
    kp, des = surf.detectAndCompute(imgGray, None)
    return des

def buildDictionary(rootPath):
    surfDes = np.empty([1,64])
    for path, subdirs, files in os.walk(rootPath):
        for filename in files:
            if filename.endswith("_thumb.jpg"):
                filename = os.path.join(path, filename)
                surfDesRow = findSurfDescriptor(filename)
                surfDes = np.vstack((surfDes, surfDesRow))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(np.float32(surfDes), 100, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    del surfDes
    gc.collect()
    return center

def main():
    root = "/Users/qtc746/Documents/Courses/ComputerVision/Project/Dataset/Dictionary/"
    center = buildDictionary(root)

if __name__ == "__main__":
    main()
