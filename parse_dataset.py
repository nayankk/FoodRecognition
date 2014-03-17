#!/usr/bin/python

import os
import math
import random


def buildTrainAndTestFiles(rootDir, isRestaurant):
    trainFiles = []
    trainLabels = []
    testFiles = []
    testLabels = []
    dictionaryfiles = []

    restaurentNames = os.listdir(rootDir)
    if isRestaurant == True:
        folder = "restaurant"
    else:
        folder = "still"

    for name in restaurentNames:
        if os.path.isdir(rootDir + "/" + name):
            items = os.listdir(rootDir + "/" + name)
            for item in items:
                if os.path.isdir(rootDir + "/" + name + "/" + item):
                    totalSet = len(os.listdir(rootDir + "/" + name + "/" + item + "/images/" + folder + "/"))
                    trainSetLen = int(round(totalSet * 3.0/4))
                    testSetLen = totalSet - trainSetLen
                    first = ''
                    firsttemp = ''
                    last = ''
                    for i in range(trainSetLen):
                        images = os.listdir(rootDir + "/" + name + "/" + item + "/images/" + folder + "/inst " + str(i+1) + "/")
                        for image in images:
                            if image.endswith("thumb.jpg"):
                                trainFiles.append(rootDir + "/" + name + "/" + item + "/images/" + folder + "/inst " + str(i+1) + "/" + image)
                                trainLabels.append(int(name + item))
                                # Try to use second image as dictionary image
                                if not firsttemp:
                                    firsttemp = rootDir + "/" + name + "/" + item + "/images/" + folder + "/inst " + str(i+1) + "/" + image
                                if images.index(image) not in (0,1):
                                    if not first:
                                        first = rootDir + "/" + name + "/" + item + "/images/" + folder + "/inst " + str(i+1) + "/" + image
                                last = rootDir + "/" + name + "/" + item + "/images/" + folder + "/inst " + str(i+1) + "/" + image
                    if not first:
                        first = fisttemp

                    tempList = []
                    tempList.append(first)
                    tempList.append(last)
                    dictionaryfiles.append(tempList)

                    for i in range(testSetLen):
                        images = os.listdir(rootDir + "/" + name + "/" + item + "/images/" + folder + "/inst " + str(trainSetLen+i+1) + "/")
                        for image in images:
                            if image.endswith("thumb.jpg"):
                                testFiles.append(rootDir + "/" + name + "/" + item + "/images/" + folder + "/inst " + str(trainSetLen+i+1) + "/" + image)
                                testLabels.append(int(name + item))

    return trainFiles, trainLabels, testFiles, testLabels, dictionaryfiles
