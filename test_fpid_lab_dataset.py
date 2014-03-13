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

    model = svmutil.svm_train(trainingLabels, trainingData, '-s 0 -t 0 -g 1 -c 100')

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
    #rootDir = "/Users/qtc746/Documents/Courses/ComputerVision/FPID_Lab_Stills"
    trainfiles, trainlabels, testfiles, testlabels, dictionaryfiles = parseDataset.buildTrainAndTestFiles(rootDir)
    dictionary = DictionaryBuilder.buildDictionaryFromFiles(dictionaryfiles, 200)
    model = train(trainlabels, trainfiles, dictionary, 200)
    test(testlabels, testfiles, model, dictionary, 200)

if __name__ == "__main__":
    main()
