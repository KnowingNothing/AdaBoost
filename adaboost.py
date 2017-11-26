#coding=utf-8
'''
***** adaboost *****
author : zsz
email  : zhengsz@pku.edu.cn
last modify: 2017-11-26
description: a simple impletation of adaboost method
copyright: the codes following are free, hope these codes can be helpful
***** ******** *****
'''

import numpy as np
# import dataOp as dt
import linearClassifier as lc
import svm
# import random as rd
# import matplotlib.pyplot as plt

# the class definition of adaboost with two methods: fit and predict
# the adaboost classifier is only capable to work in two-cluster problem
class Adaboost:
    def __init__(self, name = 'adaboost'):
        self.name = name

    # use fit method to learn from data
    def fit(self, dataX, dataY, T = 10, method = 'linear'):
        self.N = len(dataX)
        # the weights are assigned to training data
        self.weight = np.ones(self.N)/self.N
        self.H = []
        self.A = []

        # we hope to get T classifiers in total
        for i in range(T):
            label = []
            if method == 'linear':
                classifier = lc.linearClassifier()
                # use simple linear classifier
                classifier.fit(dataX=dataX, dataY=dataY, weight=self.weight)
            elif method == 'non-linear':
                classifier = svm.Svm()
                # user svm with rbf kernel
                classifier.fit(x_data=dataX, y_data=dataY, kernel=['rbf', 1])
            count = 0
            # calculate the error rate to determine the weight of this classifier
            for j in range(self.N):
                if np.sign(classifier.predict(dataX[j,:])) != dataY[j,:]:
                    count = count + 1
            error = float(count) / self.N
            # print('current error is {}'.format(error))
            # when it comes to a perfect classifier, just put it in with high weight and break
            if error == 0:
                self.A.append(9999)
                self.H.append(classifier)
                break
            if error == 1:
                continue
            # this is the weight of current classifier
            a = 0.5 * np.log(1 / error - 1)
            self.A.append(a)
            self.H.append(classifier)
            Z = 0
            # update the weights assigned to training data
            for j in range(self.N):
                self.weight[j] = self.weight[j] * np.exp(-dataY[j,:] * a * classifier.predict(dataX[j,:]))
                Z = Z + self.weight[j]
            for j in range(self.N):
                self.weight[j] = self.weight[j] / Z
        # the total number of classifiers in reality
        self.T = len(self.H)

    # use this function to predict
    def predict(self, x):
        result = 0
        for i in range(self.T):
            tmp = self.H[i].predict(x)
            tmp = tmp * self.A[i]
            result = result + tmp
        return np.sign(result)

# the following are my test codes with a hard data which is nearly not dividable
'''
def main():
    # get data from raw file
    dataX, dataY = dt.fetchData('sample.txt', 0, 56, 56, 1)

    #blues = dataX[:500,:]
    #reds = dataX[500:,:]
    #plt.scatter(reds[:, 0], reds[:, 1], c='r')
    #plt.scatter(blues[:,0], blues[:,1], c='b')

    #plt.show()
    N = len(dataX)
    # we want to shuffle it
    index = np.arange(N)
    rd.shuffle(index)
    dataX, dataY = dataX[index], dataY[index]
    # separate 80% of them to train
    trainNum = int(0.8 * N)
    testNum = N - trainNum
    adaboost = Adaboost()
    train_dataX, train_dataY = dataX[:trainNum], dataY[:trainNum]
    test_dataX, test_dataY = dataX[trainNum:], dataY[trainNum:]
    # here begins the training
    adaboost.fit(train_dataX, train_dataY, T=50, method='non-linear')

    # use the test data to verify the efficiency of our classifier
    count = 0
    for i in range(testNum):
        predict = adaboost.predict(test_dataX[i,:])
        if predict != test_dataY[i, :]:
            count = count + 1

    print('error rate on test data is {}'.format(float(count) / testNum))

if __name__ == '__main__':
    main()
'''

