import numpy as np
import matplotlib.pyplot as plt
import random

#function of predict

def predict(row,weights,delta):
    activation = delta 
    for i in range(len(row)-1):
        activation+=weights[i]*row[i]
    return 1.0 if activation>=0.0 else 0.0

#train the model

def train_we(train,lr,n_epoch):
    weights=[0.0 for i in range(len(train[0])-1)]
    for epoch in range(n_epoch):
        delta= 0.0 #bias
        sum_error=0.0 #summary error
        for row in train:
            prediction = predict(row,weights,delta)
            error = row[-1]-prediction #check prediction
            sum_error +=error ** 2
            delta = delta + lr *error
            for i in range(len(row)-1):
                weights[i]+=lr*error*row[i]
    return weights, delta #list of weights, deltas

#random points
N = 10
dataset = [[0]*4]*N

x1 = np.random.random(N)
x2 = x1 + [np.random.randint(10)/10 for i in range(N)]
x3 = np.random.random(N)
x4 = x3 - [np.random.randint(10)/10 for i in range(N)] - 0.1

for i in range(10):
    if i < 5:
        dataset[i] = [1, x1[i], x2[i], 0]
    else:
        dataset[i] = [1, x3[i], x4[i], 1]
        
l_rate=0.1 #step of bias
n_epoch=1000 #number of epochs to train
weights, delta =train_we(dataset,l_rate,n_epoch) #final weight and bias

#compare the prediction with what we have chosen
for row in dataset:
    predictit= predict(row,weights,delta)
    print('selected:%d , predicted:%d ' % (row[-1],predictit))

#dividing points into 2 groups and creating a graph
for row in dataset:
    if row[-1] == 1:
        plt.scatter(row[1],row[2],s=10, c='red')
    else:
        plt.scatter(row[1],row[2],s=10, c='blue')

#line separating the classes
def line(x1,weights):
    return ((weights[1]*x1)/-weights[2])-(weights[0]/weights[2])

#plots a line using the given weights on a graph
x1_range = np.arange(-5,5,0.5)
plt.plot(x1_range, line(x1_range, weights), color='black')
plt.grid(True)
plt.show()


        


