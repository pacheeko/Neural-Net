# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:05:01 2021

@author: bpach
"""

import numpy as np
import idx2numpy
import network
import pickle

#####################################################################################################
# testing data

with np.load("data/notMNIST.npz", allow_pickle=True) as f:
    train_features, train_labels = f['x_train'], f['y_train']
    test_features, test_labels = f['x_test'], f['y_test']

def prepTestData():
    ntest = len(test_labels)
    
    np_test_features = np.array(test_features)
    flat_test_features = np.empty((ntest, 784, 1))

    it = 0
    while it < ntest:
        flat_test_features[it] = np_test_features[it].reshape((784,1))
        it += 1
        
    #Normalize data in feature vectors   
    normal_test_features = np.empty((ntest, 784, 1))

    it = 0
    while it < ntest:
        arr = flat_test_features[it]   
        norm = np.linalg.norm(arr)
        arr = np.divide(arr, norm)
        normal_test_features[it] = arr
        it += 1
    
    it = 0
    testingData = []
    while it < ntest:
        testingData.append((normal_test_features[it], test_labels[it]))
        it += 1
    
    return ntest, testingData

#################################################################################################################

filename = 'part2.pkl'

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
