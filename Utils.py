# Utility functions

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

# Given a array of data, plot a histogram
def plotHistogram(data):
    plt.hist(data,50)
    plt.show()


# Given image and keypoints, plot keypoint over image
def plotKeypoints(imgPath, kp):
    img = cv2.imread(imgPath)
    imgShow = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
    plt.imshow(imgShow)
    plt.show()


