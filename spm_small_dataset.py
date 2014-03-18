#!/usr/bin/python

import sys
import os
import argparse
import cv2
import numpy as np
import gc
import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 

def parse_small_dataset(root_folder):
    train_file_list = []
    train_labels = []
    dictionary_files = []
    test_file_list = []
    test_labels = []
    labels = ["Burger","Drink","Pizza","Salad","Sandwich","Sub"]
    training_root_dir = root_folder + "/" + "Training" + "/"
    for path, subdirs, files in os.walk(training_root_dir):
        for filename in files:
            if filename.endswith("_thumb.jpg"):
                file_full_path = os.path.join(path, filename)
                train_file_list.append(file_full_path)
                c = file_full_path.count('/')
                train_labels.append(labels.index(os.path.dirname(file_full_path).rsplit('/')[c-1]))

    testing_root_dir = root_folder + "/" + "Testing" + "/"
    for path, subdirs, files in os.walk(testing_root_dir):
        for filename in files:
            if filename.endswith("_thumb.jpg"):
                file_full_path = os.path.join(path, filename)
                test_file_list.append(file_full_path)
                c = file_full_path.count('/')
                test_labels.append(labels.index(os.path.dirname(file_full_path).rsplit('/')[c-1]))

    dictionary_root_dir = root_folder + "/" + "Dictionary" + "/"
    for path, subdirs, files in os.walk(dictionary_root_dir):
        for filename in files:
            if filename.endswith("_thumb.jpg"):
                file_full_path = os.path.join(path, filename)
                dictionary_files.append(file_full_path)

    return train_file_list, train_labels, test_file_list, test_labels, dictionary_files

def find_surf_descriptor(filename, is_extended):
    img = cv2.imread(filename)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.SURF(hessianThreshold=100, extended=is_extended)
    kp, des = surf.detectAndCompute(imgGray, None)
    return des

def build_dictionary(dictionary_files, dictionary_size, is_extended):
    surf_des = []
    for filename in dictionary_files:
        surf_desc_row = find_surf_descriptor(filename, is_extended)
        if len(surf_des) == 0:
            surf_des = surf_desc_row
        else:
            surf_des = np.vstack((surf_des, surf_desc_row))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(np.float32(surf_des), int(dictionary_size), criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    del surf_des
    gc.collect()
    return center

def slice_image(filename):
    slices = []
    temp_folder = "/Users/qtc746/Documents/Courses/ComputerVision/Project/temp"
    img = Image.open(filename)
    width, height = img.size

    left = 0
    top = 0
    right = width/2
    bottom = height/2
    bbox = (left, top, right, bottom)
    new_img = img.crop(bbox)
    filename1 = temp_folder + "/" + filename.split('/')[filename.count('/')].strip('.jpg') + "_" + "1.jpg"
    new_img.save(filename1)

    left = 0
    top = height/2
    right = width/2
    bottom = height
    bbox = (left, top, right, bottom)
    new_img = img.crop(bbox)
    filename2 = temp_folder + "/" + filename.split('/')[filename.count('/')].strip('.jpg') + "_" + "2.jpg"
    new_img.save(filename2)

    left = width/2
    top = height/2
    right = width
    bottom = height
    bbox = (left, top, right, bottom)
    new_img = img.crop(bbox)
    filename3 = temp_folder + "/" + filename.split('/')[filename.count('/')].strip('.jpg') + "_" + "3.jpg"
    new_img.save(filename3)

    left = width/2
    top = 0
    right = width
    bottom = height/2
    bbox = (left, top, right, bottom)
    new_img = img.crop(bbox)
    filename4 = temp_folder + "/" + filename.split('/')[filename.count('/')].strip('.jpg') + "_" + "4.jpg"
    new_img.save(filename4)
    
    sliced_names = []
    sliced_names.append(filename1)
    sliced_names.append(filename2)
    sliced_names.append(filename3)
    sliced_names.append(filename4)

    return sliced_names

def get_histogram(filename, is_extended, dictionary):
    size = dictionary.shape[0]
    histogram = [0] * size
    des = find_surf_descriptor(filename, is_extended)
    if des != None:
        for one_feature in des:
            min = float("inf")
            for index, one_word in enumerate(dictionary):
                distance = cv2.norm(one_feature - one_word)
                if distance < min:
                    min = distance
                    best_match = index
            histogram[best_match] = histogram[best_match] + 1
    return histogram

def get_normalized_histogram(filename, is_extended, dictionary, spm_levels):
    histogram = []
    histogram0 = get_histogram(filename, is_extended, dictionary)
    histogram0 = [float(x) * 1/4 for x in histogram0]
    histogram.extend(histogram0)

    slices0 = slice_image(filename)
    for slice0 in slices0:
        histogram1 = get_histogram(slice0, is_extended, dictionary)
        histogram1 = [float(x) * 1/4 for x in histogram1]
        histogram.extend(histogram1)
        slices1 = slice_image(slice0)
        for slice1 in slices1:
            histogram2 = get_histogram(slice1, is_extended, dictionary)
            histogram2 = [float(x) * 1/2 for x in histogram2]
            histogram.extend(histogram2)
                
    # Normalize histogram
    total = sum(histogram)
    for i in range(len(histogram)):
        histogram[i] = float(histogram[i])/total

    return histogram

def spatial_pyramid_kernel(M, N):
    m = M.shape[0]
    n = N.shape[0]

    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp

    return result

def train(dictionary, train_file_list, train_labels, is_extended, spm_levels):
    training_data = []
    for filename in train_file_list:
        training_data.append(get_normalized_histogram(filename, is_extended, dictionary, spm_levels))
    model = SVC(C=120, kernel=spatial_pyramid_kernel, gamma=1)
    model.fit(training_data, train_labels)
    result = model.predict(training_data)
    acc = accuracy_score(result, train_labels)
    print "Accuracy = ", acc * 100
    return  model

def test(dictionary, test_file_list, test_labels, is_extended, model, spm_levels):
    testing_data = []
    for filename in test_file_list:
        testing_data.append(get_normalized_histogram(filename, is_extended, dictionary, spm_levels))
    result = model.predict(testing_data)
    acc = accuracy_score(result, test_labels)
    print acc * 100

def spm_classification(dictionary_size, spm_levels, is_extended, root_folder):
    train_file_list, train_labels, test_file_list, test_labels, dictionary_files = parse_small_dataset(root_folder)
    dictionary = build_dictionary(dictionary_files, dictionary_size, is_extended)
    print "Dictionary built", dictionary.shape
    print "Now traning.."
    model = train(dictionary, train_file_list, train_labels, is_extended, spm_levels)
    print "Now testing.."
    test(dictionary, test_file_list, test_labels, is_extended, model, spm_levels)
    
def main():
    print "SPM based classification scheme"
    parser = argparse.ArgumentParser(description='SPM based classification')
    parser.add_argument('-k', help="Dictionary size", default='100')
    parser.add_argument('-l', help="Number of SPM levels", default='3')
    parser.add_argument('-r', help="Root folder", default="/Users/qtc746/Documents/Courses/ComputerVision/Project/Dataset")
    parser.add_argument('-x', help="Use 128 length descriptors?", default=0)
    args = parser.parse_args()
    dictionary_size = args.__dict__['k']
    spm_levels = args.__dict__['l']
    root_folder = args.__dict__['r']
    is_extended = args.__dict__['x']
    spm_classification(dictionary_size, spm_levels, int(is_extended), root_folder)

if __name__ == "__main__":
    main()
