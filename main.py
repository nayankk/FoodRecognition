#!/usr/bin/python

# Main routine to train and test

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import DictionaryBuilder

categories = {}
categories["Drink"] = 1
categories["Sub"] = 2
categories["Pizza"] = 3
categories["Salad"] = 4
categories["Sandwich"] = 5

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
                histogram = [0] * 50
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

    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                       svm_type = cv2.SVM_C_SVC,
                       C=80, gamma=1 )
    svm = cv2.SVM()
    svm.train(np.float32(trainingData), np.array(labels), params=svm_params)

    # Testing
    print "Testing.."
    result = svm.predict_all(np.float32(trainingData))

    # Check accuracy on trained labels
    mask = result.reshape(-1)==labels
    correct = np.count_nonzero(mask)
    print correct*100.0/result.size

    return svm

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
                histogram = [0] * 50
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

    result = model.predict_all(np.float32(testingData))
    # Check accuracy on trained labels
    mask = result.reshape(-1)==labels
    correct = np.count_nonzero(mask)
    print correct*100.0/result.size

def main():
    root = "/Users/qtc746/Documents/Courses/ComputerVision/Project/Dataset/Dictionary/"
    dictionary = DictionaryBuilder.buildDictionary(root)
    print "Dictionary built",
    print dictionary.shape
    trainingDataRoot = "/Users/qtc746/Documents/Courses/ComputerVision/Project/Dataset/Training"
    model = train(trainingDataRoot, dictionary)
    testingDataRoot = "/Users/qtc746/Documents/Courses/ComputerVision/Project/Dataset/Testing"
    test(testingDataRoot, dictionary, model)

if __name__ == "__main__":
    main()
