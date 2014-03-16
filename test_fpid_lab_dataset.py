#!/usr/bin/python

import os
import parseDataset
import DictionaryBuilder
import cv2
import svm
import svmutil

def train(trainingLabels, trainingFiles, dictionary, k):
    print "Training.."
    trainingData = []
    for filename in trainingFiles:
        # Build histogram
        histogram = [0] * k
        des = DictionaryBuilder.findSurfDescriptor(filename)
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

    model = svmutil.svm_train(trainingLabels, trainingData, '-s 0 -t 0 -g 1 -c 50')

    # Testing
    result, acc, vals = svmutil.svm_predict(trainingLabels, trainingData, model)
    print acc

    svmutil.svm_save_model("mymodel.model", model)

    return model

def test(testingLabels, testingFiles, model, dictionary, k):
    print "Testing..."
    testingData = []

    for filename in testingFiles:
        # Build histogram
        histogram = [0] * k
        des = DictionaryBuilder.findSurfDescriptor(filename)
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
    trainfiles1, trainlabels1, testfiles1, testlabels1, dictionaryfiles1 = parseDataset.buildTrainAndTestFiles(rootDir, True)
    rootDir = "/Users/qtc746/Documents/Courses/ComputerVision/FPID_Lab_Stills"
    trainfiles2, trainlabels2, testfiles2, testlabels2, dictionaryfiles2 = parseDataset.buildTrainAndTestFiles(rootDir, False)
    trainfiles = trainfiles1 + trainfiles2
    trainlabels = trainlabels1 + trainlabels2
    testfiles = testfiles1 + testfiles2
    testlabels = testlabels1 + testlabels2
    dictionaryfiles = dictionaryfiles1 + dictionaryfiles2

    print "Total categories = ", len(set(trainfiles))
    print "Total train files = ", len(trainfiles), len(trainlabels)
    print "Total test files = ", len(testfiles), len(testlabels)
    print "Total dictionary files = ", len(dictionaryfiles)

    k = 256
    dictionary = DictionaryBuilder.buildDictionaryFromFiles(dictionaryfiles, k)
    model = train(trainlabels, trainfiles, dictionary, k)
    test(testlabels, testfiles, model, dictionary, k)

if __name__ == "__main__":
    main()
