import numpy as np
import network
import pickle

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
    
MAX_SIZE = 255
#################################################################

# reads the data from the notMNIST.npz file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # loads the four arrays specified.
    # train_features and test_features are arrays of (28x28) pixel values from 0 to 255.0
    # train_labels and test_labels are integers from 0 to 9 inclusive, representing the letters A-J
    with np.load("data/notMNIST.npz", allow_pickle=True) as f:
        train_features, train_labels = f['x_train'], f['y_train']
        test_features, test_labels = f['x_test'], f['y_test']
        
    # need to rescale, flatten, convert training labels to one-hot, and zip appropriate components together
    
    #flatten training and testing features
    ntrain = len(train_labels)
    ntest = len(test_labels)
    
    np_train_features = np.array(train_features)
    np_test_features = np.array(test_features)
    flat_train_features = np.empty((ntrain, 784, 1)) 
    flat_test_features = np.empty((ntest, 784, 1))
    it = 0
    while it < ntrain:
        flat_train_features[it] = np_train_features[it].reshape((784,1))
        it += 1
        
    it = 0
    while it < ntest:
        flat_test_features[it] = np_test_features[it].reshape((784,1))
        it += 1
        
    
    #Normalize data in feature vectors   
    normal_train_features = np.empty((ntrain, 784, 1))
    normal_test_features = np.empty((ntest, 784, 1))
    
    it = 0
    while it < ntrain:
        arr = flat_train_features[it]
        norm = np.linalg.norm(arr)
        arr = np.divide(arr, norm)
        normal_train_features[it] = arr
        it += 1

    it = 0
    while it < ntest:
        arr = flat_test_features[it]   
        norm = np.linalg.norm(arr)
        arr = np.divide(arr, norm)
        normal_test_features[it] = arr
        it += 1
    
    #Convert training labels to one hot
    train_labels_oh = np.empty((ntrain, 10, 1))
    it = 0
    while it < ntrain:
        train_labels_oh[it] = onehot(train_labels[it] ,10)
        it += 1
    
    trainingData = []
    testingData = []
    
    it = 0
    while it < ntrain:
        trainingData.append((normal_train_features[it], train_labels_oh[it]))
        it += 1
        
    it = 0
    while it < ntest:
        testingData.append((normal_test_features[it], test_labels[it]))
        it += 1
    
    return (trainingData, testingData)
    
###################################################################

trainingData, testingData = prepData()

net = network.Network([784,20,10])
net.SGD(trainingData, 30, 20, 20, test_data = testingData)
pickle.dump(net, open('part2.pkl', 'wb'))







