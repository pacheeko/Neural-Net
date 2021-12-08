# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:40:24 2021

@author: bpach
"""

import numpy as np
import idx2numpy
import network
import pickle

#####################################################################################################
# testing data
testingImageFile = "data/t10k-images.idx3-ubyte"
testingLabelFile = "data/t10k-labels.idx1-ubyte"


# returns the number of entries in the file, as well as a list of integers
# representing the correct label for each entry
def getLabels(labelfile):
    file = open(labelfile, 'rb')
    file.read(4)
    n = int.from_bytes(file.read(4), byteorder='big') # number of entries
    
    labelarray = bytearray(file.read())
    labelarray = [b for b in labelarray]    # convert to ints
    file.close()
    
    return n, labelarray

# returns a list containing the pixels for each image, stored as a (784, 1) numpy array
def getImgData(imagefile):
    # returns an array whose entries are each (28x28) pixel arrays with values from 0 to 255.0
    images = idx2numpy.convert_from_file(imagefile) 
    
    # We want to flatten each image from a 28 x 28 to a 784 x 1 numpy array
    img_len = len(images)
    it = 0
    np_images = np.array(images)
    flat_images = np.empty((img_len, 784, 1))
    while it < img_len:
        flat_images[it] = np_images[it].reshape((784,1))
        it += 1
    
    # convert to floats in [0,1] (only really necessary if you have other features, but we'll do it anyways)
    it = 0
    normal_images = np.empty((img_len, 784, 1))
    #inner_length = 784
    while it < img_len:
        arr = flat_images[it]
        norm = np.linalg.norm(arr)
        arr = np.divide(arr, norm)
        normal_images[it] = arr
        #it2 = 0
        #while it2 < 784:
         #   print(flat_images[it][it2])
          #  flat_images[it][it2] = standardize(flat_images[it][it2], mean, stdDev)
           # print(flat_images[it][it2])
            #it2 += 1
        it += 1
    return normal_images

def prepTestData():
    ntest, test_labels = getLabels(testingLabelFile)

    test_images = getImgData(testingImageFile)
    
    it = 0
    testingData = []
    while it < ntest:
        testingData.append((test_images[it], test_labels[it]))
        it += 1
    
    return ntest, testingData

#################################################################################################################

filename = 'part1.pkl'

with open(filename, 'rb') as f:
    net = pickle.load(f)
    
ntest, test_data = prepTestData()

test_results = [(np.argmax(net.feedforward(x)), y) for (x, y) in test_data]

length = len(test_results)
it = 0
num = 0
for (x,y) in test_results:
    if (x != y):
        print("Index: " + str(it) + " Guess: " + str(x) + " Actual: " + str(y))
        num += 1
    it += 1

print("total failed: " + str(num))
