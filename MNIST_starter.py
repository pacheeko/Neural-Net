import numpy as np
import idx2numpy
import network
import pickle
import datetime

# converts a 1d python list into a (1,n) row vector
def rv(vec):
    return np.array([vec])
    
# converts a 1d python list into a (n,1) column vector
def cv(vec):
    return rv(vec).T
        
# creates a (size,1) array of zeros, whose ith entry is equal to 1    
def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)

# **Code taken from lecture**
# Standardizes the array to values between 0 and 1
def standardize(x, mu, sigma):
    return (x - mu) / sigma
    
##################################################
# NOTE: make sure these paths are correct for your directory structure

# training data
trainingImageFile = "data/train-images.idx3-ubyte"
trainingLabelFile = "data/train-labels.idx1-ubyte"

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

# reads the data from the four MNIST files,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    ntrain, train_labels = getLabels(trainingLabelFile)
    train_labels_oh = np.empty((ntrain, 10,1))
    it = 0
    while it < ntrain:
        train_labels_oh[it] = onehot(train_labels[it],10)
        it += 1
    
    ntest, test_labels = getLabels(testingLabelFile)
    train_images = getImgData(trainingImageFile)
    test_images = getImgData(testingImageFile)
    
    it = 0;
    trainingData = []
    while it < ntrain:
        trainingData.append((train_images[it], train_labels_oh[it]))
        it += 1
        
    it = 0
    testingData = []
    while it < ntest:
        testingData.append((test_images[it], test_labels[it]))
        it += 1
    
    return (trainingData, testingData)
    

###################################################

trainingData, testingData = prepData()
net = network.Network([784,20,10])
start = datetime.datetime.now()
net.SGD(trainingData, 20, 20, 5, test_data = testingData)
end = datetime.datetime.now()
diff = end - start
print("Training time: " + str(diff.seconds) + " seconds, " + str(diff.microseconds) + " microseconds")
#pickle.dump(net, open('part1.pkl', 'wb'))






        