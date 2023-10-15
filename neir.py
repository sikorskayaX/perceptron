import numpy as np
import matplotlib.pyplot as plt
import random
import copy

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

#create dataset to save points
N = 20 
dataset = [[0]*4]*N
dataset1 = [[0]*4]*N
dataset2 = [[0]*4]*N


# points for class 0
x1_0 = np.random.uniform(0.5 , 1.5, N)
x2_0 = np.random.uniform(0.5 , 1.5, N)

# points for class 1
x1_1 = np.random.uniform(-0.5 , -1.5, N)
x2_1 = np.random.uniform(-0.5 , -1.5, N)

# points for class 2
x1_2 = np.random.uniform(-0.5 , -1.5, N)
x2_2 = np.random.uniform(0.5 , 1.5, N)

# points for class 3
x1_3 = np.random.uniform(0.5 , 1.5, N)
x2_3 = np.random.uniform(-0.5 , -1.5, N)

# dividing points into 4 classes
for i in range(N):
    if i < 5:
        dataset[i] = [1, x1_0[i], x2_0[i], 0]
    elif (i >= 5) and ( i < 10):
        dataset[i] = [1, x1_1[i], x2_1[i], 1]
    elif (i >= 10) and ( i < 15):
        dataset[i] = [1, x1_2[i], x2_2[i], 2]
    else:
        dataset[i] = [1, x1_3[i], x2_3[i], 3] 
        
        
l_rate=0.1 #step of bias
n_epoch=1000 #number of epochs to train
weights, delta =train_we(dataset,l_rate,n_epoch) #final weight and bias

#copy
dataset1  = copy.deepcopy(dataset)
dataset2  = copy.deepcopy(dataset)

 
#combining 0 and 4 classes
for i in range (N):
    if (i < 5) or (i >= 15):
        dataset1[i][3] = 0
    else:
        dataset1[i][3] = 1
    
#combining 2 and 3 classes
for i in range (N):
    if (i < 5) or(10 <= i <15):
        dataset2[i][3] = 0
    else:
        dataset2[i][3] = 1


weights_1, delta_1 =train_we(dataset1,l_rate,n_epoch) 
weights_2, delta_2 =train_we(dataset2,l_rate,n_epoch)

#compare the prediction with what we have chosen
for row in dataset:
    if row[-1] == 0:
        predictit= predict(row,weights_1,delta_1)
        print('selected:%d , predicted:%d ' % (row[-1],predictit)) 
    elif row[-1] == 1:
        predictit= predict(row,weights_1,delta_1)
        print('selected:%d , predicted:%d ' % (row[-1],predictit)) 
    elif row[-1] == 2:
        predictit= predict(row,weights_2,delta_2)+ 2
        print('selected:%d , predicted:%d ' % (row[-1],predictit)) 
    else:
        predictit= predict(row,weights_2,delta_2)+ 2
        print('selected:%d , predicted:%d ' % (row[-1],predictit))

#colors of points on graph
for row in dataset:
    if row[-1] == 0:
        plt.scatter(row[1],row[2],s=10, c='red')
    elif row[-1] == 1:
        plt.scatter(row[1],row[2],s=10, c='orange')
    elif row[-1] == 2:
        plt.scatter(row[1],row[2],s=10, c='green')
    else:
        plt.scatter(row[1],row[2],s=10, c='blue') 

#line separating the classes
def line(x1,weights):
    return ((weights[1]*x1)/-weights[2])-(weights[0]/weights[2])

#plots a line using the given weights on a graph
x1_range = np.arange(-5,5,0.5)
plt.plot(x1_range, line(x1_range, weights_1), color='black') 
plt.plot(x1_range, line(x1_range, weights_2), color='black') 
plt.grid(True)
plt.show()


        


