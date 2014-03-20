# Utility functions

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

# Given a array of data, plot a histogram
def plotHistogram(data):
    plt.hist(data,50)
    plt.show()

def findSurfDescriptor(filename):
    img = cv2.imread(filename)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.SURF(hessianThreshold=100, extended=False)
    kp, des = surf.detectAndCompute(imgGray, None)
    plotKeypoints(filename, kp)
    return des

# Given image and keypoints, plot keypoint over image
def plotKeypoints(imgPath, kp):
    img = cv2.imread(imgPath)
    imgShow = cv2.drawKeypoints(img, kp, None, (255,0,0), 0)
    plt.imshow(imgShow)
    plt.show()

def main():
    img = "/Users/qtc746/Documents/Courses/ComputerVision/FPID_Restuarant_Stills/1/11/images/restaurant/inst 1/img_1195thumb.jpg"
    findSurfDescriptor(img)
    

if __name__ == "__main__":
    main()

