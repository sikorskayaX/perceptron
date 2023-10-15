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
dataset = [[0]*4]*10
x1 = [0.03, 0.57, 0.62, 0.55, 0.67, 0.73, 0.18, 0.62, 0.16, 0.69]
x2 = [0.54, 0.32, 0.11, 0.94, 0.33, 0.04, 0.15, 0.43, 0.51, 0.18]
x3 = [0.76,  0.48, -0.28,  -0.13, 0.33, 0.35, -0.63, 0.95, -0.89, 0.03]
x4 = [ 0.03, 0.01, -0.33, 0.74, -0.47, -0.46, 0.04, -0.37,  0.31,  0.08]

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


        


