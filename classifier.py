import csv
import random
import math
import numpy as np

##Prof Eickhoff's Methods that we use
def loadCsv(filename):
    lines = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
    next(lines)
    dataset = []
    for row in lines:
        dataset.append([float(x) for x in row])
    return dataset

#Split data into training and validation portions
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


###OUR CODE (NEW!!!!)
num_epochs = 3000 # Can change this around a bit, but we tried numbers out and found this value worked
batch_size = 15 # Can change this around a bit, but we tried numbers out and found this value worked
weights = None

#helper method to run sigmoid function
def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))

#helper method to shuffle the data before creating batches
def shuffleHelper(X, shuffleArr):
    shuffledX = X[shuffleArr]
    return shuffledX

#our training method; uses batch stochastic gradient descent with weights!
def train(X, Y, weightMultiplier, learningRate):
    '''
    Train the model, using batch stochastic gradient descent
    inputs:
    X: a 2D Numpy array. Each row is an examples, each column is a feature.
    Y: a 1D Numpy array, of all the corresponding labels to the examples from X
    output:
    the updated weights Numpy array
    '''
    
    numOfExaples = X.shape[0]
    numOfFeatures = X.shape[1]
    
    #initialize weights as a bunch of 0s (length = num of features)
    weights = np.zeros(numOfFeatures)
    
    epoch_count = 0
    while (num_epochs > epoch_count): #keep looping thru til epoch counts reaches num_epochs
        
        #shuffle examples around and divide into batches based on numOfExamples / batch_size
        shuffleGuideArr = np.arange(0, numOfExaples)
        np.random.shuffle(shuffleGuideArr)

        shuffledX = shuffleHelper(X, shuffleGuideArr)
        shuffledY = shuffleHelper(Y, shuffleGuideArr)
        
        numOfSubArrays = (int) (numOfExaples / batch_size)
        batchesXY = (np.array_split(shuffledX, numOfSubArrays), np.array_split(shuffledY, numOfSubArrays))

        #for every subarray of length batch_size ...
        for j in range(numOfSubArrays):
            batchX = batchesXY[0][j]
            batchY = batchesXY[1][j]
            
            #calculate logit
            logit = np.dot(batchX, weights)
            logit = logit.astype(float)

            #find p matrix (sigmoided and learning from actual labels)
            p = sigmoid_function(logit) #sigmoid, don't need apply along axis
            p = p - batchY*weightMultiplier

            transposeBatch = batchX.transpose()
            #calculate gradient!
            lw = (np.dot(transposeBatch,p) / batch_size)
            
            #update weights based on gradient and learning rate
            weights = weights - learningRate*lw
        epoch_count = epoch_count + 1
            
    print ("done training")

    #return the final weights after all the training :)
    return weights

def predict(X, weights):
    '''
    inputs:
    features of the data (likely testing data)
    @return:
    predictions of all the examples in the input X data
    '''
    dotProd = np.dot(X, weights)
    dotProd = dotProd.astype(float)
    sigmoidedDotProd = sigmoid_function(dotProd)
        
    predicted = [1.0 if x > 0.5 else 0.0 for x in sigmoidedDotProd]
    
    return predicted

#Professor Eickhoff's method to evaluate!
def evaluate(testSet, predictions):
    tp1 = 0.0
    fp1 = 0.0
    fn1 = 0.0
    tp0 = 0.0
    fp0 = 0.0
    fn0 = 0.0
    for i in range(len(testSet)):
        if testSet[i][-1] == 0:
            if predictions[i] == 0:
                tp0 += 1.0
            else:
                fp1 += 1.0
                fn0 += 1.0
        else:
            if predictions[i] == 1:
                tp1 += 1.0
            else:
                fp0 += 1.0
                fn1 += 1.0

    p0 = tp0/(tp0+fp0)
    p1 = tp1/(tp1+fp1)
    r0 = tp0/(tp0+fn0)
    r1 = tp1/(tp1+fn1)
    print("F1 (Class 0): "+str((2.0*p0*r0)/(p0+r0)))
    print("F1 (Class 1): "+str((2.0*p1*r1)/(p1+r1)))


def main():
    #set random seed and load in data!
    random.seed(12)
    filename = 'readmission.csv'
    filename2 = 'readmission_test.csv'
    dataset = loadCsv(filename)
    testset = loadCsv(filename2)
    
    #take in and training data and divide into X and Y
    X_train = np.array(dataset)
    Y_train = X_train[:,-1]
    #X_train = np.delete(X_train, -1, 1)
    X_val = X_val[:, :68]

    #take in testing data and divide into X and Y
    X_val = np.array(testset)
    Y_val = X_val[:,-1]
    X_val = X_val[:,:68]

        
    weights = train(X_train, Y_train, 1.8, .005)
    predictions = predict(X_val, weights)
    evaluate(testset, predictions)

main()