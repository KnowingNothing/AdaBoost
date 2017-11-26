#coding=utf-8
'''
***** linearClassifier *****
author : zsz
email  : zhengsz@pku.edu.cn
last modify: 2017-11-26
description: a simple linear classifier
copyright: the codes following are free, hope these codes can be helpful
***** ******** *****
'''
import numpy as np
# import dataOp as data
# import matplotlib.pyplot as plt


class linearClassifier:
    def __init__(self, name = 'linearClassifier'):
        self.name = name

    def fit(self, dataX, dataY, weight = np.array([]), maxStep = 5000, rate = 0.01):
        dataZ = np.column_stack((dataX, np.ones_like(dataY)))
        self.W = np.zeros_like(dataZ[0])
        N = len(dataZ)
        lastW = np.ones_like(dataZ[0])
        diff = lastW - self.W
        count = 0
        while np.dot(diff, diff.T) > 1e-5 and count <= maxStep:
            lastW = self.W
            for i in range(N):
                count = count + 1
                predict = np.sign(np.dot(self.W, dataZ[i, :]))
                if predict != dataY[i,:]:
                    if len(weight) != N:
                        self.W = self.W + rate * dataY[i, :] * dataZ[i, :]
                    else:
                        self.W = self.W + rate * dataY[i, :] * dataZ[i, :] * weight[i]
            diff = lastW - self.W
            self.W = np.mat(self.W)
        return self.W

    def predict(self, x):
        x = np.mat(x)
        z = np.column_stack((x, [[1]]))
        z = np.mat(z)
        return np.sign(np.dot(self.W, z.T))


# test the efficency
'''
def main():
    x_data, y_data = data.fetchData('sample.txt', 0, 2, 3, 1)
    linear = linearClassifier()
    w = linear.fit(x_data, y_data, np.random.uniform(0,1,100))

    red = x_data[:50,:]
    blue = x_data[50:, :]
    x = np.linspace(-2,5,100)
    y = (x * w[0] + w[2]) / (-w[1])
    plt.scatter(red[:, 0], red[:, 1], c='r')
    plt.scatter(blue[:, 0], blue[:, 1], c='b')
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    main()
'''