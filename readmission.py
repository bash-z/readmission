import csv
import random
import math
import numpy as np


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

num_epochs = 3000 # Feel free to play around with this if you'd like, though this value will do
batch_size = 15 # Feel free to play around with this if you'd like, though this value will do
weights = None

def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))

def shuffleHelper(X, shuffleArr):
    shuffledX = X[shuffleArr]
    return shuffledX

def train(X, Y, weightMultiplier, learningRate):
    '''
    Train the model, using batch stochastic gradient descent
    @params:
    X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
    Y: a 1D Numpy array containing the corresponding labels for each example
    @return:
    None
    '''
    
    numOfExaples = X.shape[0]
    numOfFeatures = X.shape[1]
    
    #initialize weights as a bunch of 0s (length = num of features)
    weights = np.zeros(numOfFeatures)
    
    epoch_count = 0
    while (num_epochs > epoch_count):
        shuffleGuideArr = np.arange(0, numOfExaples)
        np.random.shuffle(shuffleGuideArr)

        shuffledX = shuffleHelper(X, shuffleGuideArr)
        shuffledY = shuffleHelper(Y, shuffleGuideArr)

        numOfSubArrays = (int) (numOfExaples / batch_size)
        batchesXY = (np.array_split(shuffledX, numOfSubArrays), np.array_split(shuffledY, numOfSubArrays))

        for j in range(numOfSubArrays):
            batchX = batchesXY[0][j]
            batchY = batchesXY[1][j]
            logit = np.dot(batchX, weights)
            logit = logit.astype(float)

            p = sigmoid_function(logit) #sigmoid, don't need apply along axis
            p = p - batchY*weightMultiplier

            transposeBatch = batchX.transpose()
            lw = (np.dot(transposeBatch,p) / batch_size)
            
            weights = weights - learningRate*lw
        epoch_count = epoch_count + 1
            
    print ("done training")

    return weights

def predict(X, weights):
    '''
    Compute predictions based on the learned parameters and examples X
    @params:
    X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
    @return:
    A 1D Numpy array with one element for each row in X containing the predicted class.
    '''
    dotProd = np.dot(X, weights)
    dotProd = dotProd.astype(float)
    sigmoidedDotProd = sigmoid_function(dotProd)
        
    predicted = [1.0 if x > 0.5 else 0.0 for x in sigmoidedDotProd]
    
    return predicted

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

def evenDistributedTraining(trainingSet):
    trainingSetNp = np.array(trainingSet)

    totalNumOf1s = sum(trainingSetNp[:,-1])
    class0 = np.empty((0,69), str)
    class1 = np.array((0,69), str)
    

    print ("Shape: ")
    print (trainingSetNp.shape) #(23088, 69)
    
    print (trainingSetNp[0])
    for i in range(trainingSetNp.shape[0]):
        if trainingSetNp[i, -1] == 0:
            class0 = np.append(class0, trainingSet[i])
        else:
            class1 = np.append(class1, trainingSet[i])
    
    shuffledClass0 = np.random.shuffle(class0)
    
    miniClass0 = shuffledClass0[0:totalNumOf1s]
    
    evenDistribTraining = np.concatenate((miniClass0, class1), axis=0)
    return evenDistribTraining






def main():
    random.seed(12)
    filename = 'readmission.csv'
    filename2 = 'readmission_test.csv'
    dataset = loadCsv(filename)
    testset = loadCsv(filename2)
    
    
    X_train = np.array(dataset)
    Y_train = X_train[:,-1]
    X_train = np.delete(X_train, -1, 1)

    X_val = np.array(testset)
    Y_val = X_val[:,-1]
    X_val = X_val[:,:68]

        
    weights = train(X_train, Y_train, 1.8, .005)
    predictions = predict(X_val, weights)
    evaluate(testset, predictions)

main()