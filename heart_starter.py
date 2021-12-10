import csv
import numpy as np
import network
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

# given a data point, mean, and standard deviation, returns the z-score
def standardize(x, mu, sigma):
    return ((x - mu)/sigma)
    

##############################################

# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def readData(filename):
    #** code taken from https://realpython.com/python-csv/ **
    features = []
    labels = []
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        n = 0
        for row in csv_reader:
            if n == 0:
                print(f'Column names are {", ".join(row)}')
                n += 1
            else:
                features.append([row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9]])
                labels.append(row[10])
                n += 1
    
    return n, features, labels


################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():

    n, features, labels = readData('data/heart.csv')
    
    n = n-1
    for f in features:
        if (f[4] == "Present"):
            f[4] = 1
        else:
            f[4] = 0
    
    arr = np.empty((n, 9, 1))
    it = 0
    while it < n:
        arr[it] = cv(features[it])
        it += 1
        
    means = np.ndarray.mean(arr, axis=0).flatten().round(2)
    stds = np.ndarray.std(arr, axis = 0).flatten().round(2)
    maxAge = np.ndarray.max(arr, axis = 0)[8][0]
    #standardize all features except family history and age. For age, divide by the max age
    for f in arr:
        f[0] = standardize(float(f[0]), float(means[0]), float(stds[0])) #sbp
        f[1] = standardize(float(f[1]), float(means[1]), float(stds[1])) #tobacco
        f[2] = standardize(float(f[2]), float(means[2]), float(stds[2])) #ldl
        f[3] = standardize(float(f[3]), float(means[3]), float(stds[3])) #adiposity                                             
        f[5] = standardize(float(f[5]), float(means[5]), float(stds[5])) #typea
        f[6] = standardize(float(f[6]), float(means[6]), float(stds[6])) #obesity
        f[7] = standardize(float(f[7]), float(means[7]), float(stds[7])) #alcohol
        f[8] = float(f[8][0])/float(maxAge) #age
    
    #Take 1/6 of the data points as test data, the rest as training data    
    ntest = n/6
    ntrain = n - ntest
    
    trainingData = []
    testingData = []
    #Add the first ntrain features and labels to the training vector.
    #Convert labels to onehot
    it = 0;
    while it < ntrain:
        label_oh = onehot(int(labels[it]),2)
        trainingData.append((arr[it],label_oh))
        it += 1
    #Add the rest of the features and labels to the test vector
    while it < n:
        testingData.append((arr[it], int(labels[it])))
        it += 1
    
    return (trainingData, testingData)


###################################################
trainingData, testingData = prepData()
net = network.Network([9,20,2])
start = datetime.datetime.now()
net.SGD(trainingData, 100, 30, 0.3, test_data = testingData)
end = datetime.datetime.now()
diff = end - start
print("Training time: " + str(diff.seconds) + " seconds, " + str(diff.microseconds) + " microseconds")


       