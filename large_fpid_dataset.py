#!/usr/bin/python

import os
import parse_dataset
import cv2
import svm
import svmutil
import numpy as np

def findSurfDescriptor(filename):
    img = cv2.imread(filename)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.SURF(hessianThreshold=100, extended=False)
    kp, des = surf.detectAndCompute(imgGray, None)
    return des

def buildDictionaryFromFiles(dictionaryfiles, k):
    surfDes = np.empty([1,64])
    for filenameList in dictionaryfiles:
        for filename in filenameList:
            surfDesRow = findSurfDescriptor(filename)
            if surfDesRow is None:
                continue
            if len(surfDes) == 1:
                surfDes = surfDesRow
            else:
                surfDes = np.vstack((surfDes, surfDesRow))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(np.float32(surfDes), k, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    print "Dictionary built, size = ", center.shape
    return center

def train(trainingLabels, trainingFiles, dictionary, k):
    print "Training.."
    trainingData = []
    for filename in trainingFiles:
        # Build histogram
        histogram = [0] * k
        des = findSurfDescriptor(filename)
        if des is None:
            trainingData.append(histogram)
            continue
        for oneFeature in des:
            min = float("inf")
            for index, oneWord in enumerate(dictionary):
                distance = cv2.norm(oneFeature - oneWord)
                if distance < min:
                    min = distance
                    bestMatch = index
            histogram[bestMatch] = histogram[bestMatch] + 1
                
        # Normalize histogram
        total = sum(histogram)
        for i in range(len(histogram)):
            histogram[i] = float(histogram[i])/total

        # Add it to training data
        trainingData.append(histogram)

    model = svmutil.svm_train(trainingLabels, trainingData, '-s 0 -t 0 -g 1 -c 120')

    # Testing
    result, acc, vals = svmutil.svm_predict(trainingLabels, trainingData, model)
    print acc

    return model

def test(testingLabels, testingFiles, model, dictionary, k):
    print "Testing..."
    testingData = []

    for filename in testingFiles:
        # Build histogram
        histogram = [0] * k
        des = findSurfDescriptor(filename)
        if des is None:
            testingData.append(histogram)
            continue
        for oneFeature in des:
            min = float("inf")
            for index, oneWord in enumerate(dictionary):
                distance = cv2.norm(oneFeature - oneWord)
                if distance < min:
                    min = distance
                    bestMatch = index
            histogram[bestMatch] = histogram[bestMatch] + 1
                
        # Normalize histogram
        total = sum(histogram)
        for i in range(len(histogram)):
            histogram[i] = float(histogram[i])/total

        # Add it to training data
        testingData.append(histogram)

    result, acc, vals = svmutil.svm_predict(testingLabels, testingData, model)
    print acc

def main():
    print "Starting to build train set and test set"
    rootDir = "/Users/qtc746/Documents/Courses/ComputerVision/FPID_Restuarant_Stills"
    trainfiles1, trainlabels1, testfiles1, testlabels1, dictionaryfiles1 = parse_dataset.buildTrainAndTestFiles(rootDir, True)
    rootDir = "/Users/qtc746/Documents/Courses/ComputerVision/FPID_Lab_Stills"
    trainfiles2, trainlabels2, testfiles2, testlabels2, dictionaryfiles2 = parse_dataset.buildTrainAndTestFiles(rootDir, False)
    trainfiles = trainfiles1 + trainfiles2
    trainlabels = trainlabels1 + trainlabels2
    testfiles = testfiles1 + testfiles2
    testlabels = testlabels1 + testlabels2
    dictionaryfiles = dictionaryfiles1 + dictionaryfiles2

    print "Total categories = ", len(set(trainlabels))
    print "Total train files = ", len(trainfiles)
    print "Total test files = ", len(testfiles)
    print "Total dictionary files = ", len(dictionaryfiles)

    k = 256
    dictionary = buildDictionaryFromFiles(dictionaryfiles, k)
    model = train(trainlabels, trainfiles, dictionary, k)
    test(testlabels, testfiles, model, dictionary, k)

if __name__ == "__main__":
    main()
