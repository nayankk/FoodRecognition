#!/usr/bin/python

# Main routine to train and test

import cv2
import numpy as np
import os
import gc
import svm
import svmutil
import DictionaryBuilder

categories = {}
categories["Drink"] = 1
categories["Sub"] = 2
categories["Pizza"] = 3
categories["Salad"] = 4
categories["Sandwich"] = 5
categories["Burger"] = 6

def train(trainingRootDir, dictionary):
    print "Training.."
    labels = []
    trainingData = []

    for path, subdirs, files in os.walk(trainingRootDir):
        for filename in files:
            if filename.endswith("_thumb.jpg"):
                filename = os.path.join(path, filename)
                # Get labels
                (head, tail) = os.path.split(path)
                labels.append(categories[tail])

                # Build histogram
                histogram = [0] * 100
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

    model = svmutil.svm_train(labels, trainingData, '-s 0 -t 0 -g 1 -c 100')

    # Testing
    result, acc, vals = svmutil.svm_predict(labels, trainingData, model)
    print acc

    svmutil.svm_save_model("mymodel.model", model)

    return model

def test(testingRootDir, dictionary, model):
    print "Testing..."
    labels = []
    testingData = []

    for path, subdirs, files in os.walk(testingRootDir):
        for filename in files:
            if filename.endswith("_thumb.jpg"):
                filename = os.path.join(path, filename)
                # Get labels
                (head, tail) = os.path.split(path)
                labels.append(categories[tail])

                # Build histogram
                histogram = [0] * 100
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

    result, acc, vals = svmutil.svm_predict(labels, testingData, model)
    print acc

def main():
    root = "/Users/qtc746/Documents/Courses/ComputerVision/Project/Dataset/Dictionary/"
    print "Building dictionary"
    dictionary = DictionaryBuilder.buildDictionary(root)
    print "Dictionary built",
    print dictionary.shape
    trainingDataRoot = "/Users/qtc746/Documents/Courses/ComputerVision/Project/Dataset/Training"
    model = train(trainingDataRoot, dictionary)
    testingDataRoot = "/Users/qtc746/Documents/Courses/ComputerVision/Project/Dataset/Testing"
    test(testingDataRoot, dictionary, model)

if __name__ == "__main__":
    main()
