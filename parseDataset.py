#!/usr/bin/python

import os
import math
import random

trainFiles = []
trainLabels = []
testFiles = []
testLabels = []
dictionaryfiles = []

def buildTrainAndTestFiles(rootDir):
    print "Building train and test files"
    restaurentNames = os.listdir(rootDir)
    for name in restaurentNames:
        if os.path.isdir(rootDir + "/" + name):
            items = os.listdir(rootDir + "/" + name)
            for item in items:
                if os.path.isdir(rootDir + "/" + name + "/" + item):
                    totalSet = len(os.listdir(rootDir + "/" + name + "/" + item + "/images/restaurant/"))
                    trainSetLen = int(round(totalSet * 3.0/4))
                    testSetLen = totalSet - trainSetLen
                    first = ''
                    last = ''
                    for i in range(trainSetLen):
                        images = os.listdir(rootDir + "/" + name + "/" + item + "/images/restaurant/inst " + str(i+1) + "/")
                        for image in images:
                            if image.endswith("thumb.jpg"):
                                trainFiles.append(rootDir + "/" + name + "/" + item + "/images/restaurant/inst " + str(i+1) + "/" + image)
                                trainLabels.append(int(name + item))
                                if not first:
                                    first = rootDir + "/" + name + "/" + item + "/images/restaurant/inst " + str(i+1) + "/" + image
                                last = rootDir + "/" + name + "/" + item + "/images/restaurant/inst " + str(i+1) + "/" + image

                    dictionaryfiles.append(first)
                    dictionaryfiles.append(last)

                    for i in range(testSetLen):
                        images = os.listdir(rootDir + "/" + name + "/" + item + "/images/restaurant/inst " + str(trainSetLen+i+1) + "/")
                        for image in images:
                            if image.endswith("thumb.jpg"):
                                testFiles.append(rootDir + "/" + name + "/" + item + "/images/restaurant/inst " + str(trainSetLen+i+1) + "/" + image)
                                testLabels.append(int(name + item))

    print "Train files length =", len(trainFiles)
    print "Test files length =", len(testFiles)
    print len(trainFiles)
    print len(trainLabels)
    print "Length of dictionary =", len(dictionaryfiles)
    print "Number of categories =", len(set(trainLabels))
    return trainFiles, trainLabels, testFiles, testLabels, dictionaryfiles


def main():
    print "Starting to build train set and test set"
    rootDir = "/Users/qtc746/Documents/Courses/ComputerVision/FPID_Restuarant_Stills"
    buildTrainAndTestFiles(rootDir)


if __name__ == "__main__":
    main()

