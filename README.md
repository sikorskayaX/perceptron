# perceptron ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

This code implements a simple Perceptron model for multi-class classification and builds a graph.

## clone repo
```bash
$ git clone https://github.com/sikorskayaX/perceptron.git
```
## what does this code do

Here's a step-by-step breakdown:

It first defines a predict function that calculates the weighted sum of inputs and returns 1 if the sum is greater than or equal to 0, otherwise it returns 0.

The train_we function is defined to train the model. It initializes the weights to 0 and then updates them for a specified number of epochs. The weights are updated based on the error between the predicted and actual output.

It then generates a dataset of 20 points, divided into <b>4 classes</b>. Each class is represented by a different range of x1 and x2 values.

The Perceptron model is trained on this dataset using a learning rate of 0.1 and 1000 epochs.

The dataset is then divided into two new datasets, dataset1 and dataset2. In dataset1, classes 0 and 3 are combined into a single class, and classes 1 and 2 are combined into another class. 

The model's predictions for each point in the original dataset are then printed out, along with the actual class of the point.

Finally, the points are plotted on a graph, with different colors representing different classes. The decision boundaries (lines that separate the classes) are also plotted.
## result
```bash
selected:0 , predicted:0 
selected:0 , predicted:0 
selected:0 , predicted:0 
selected:0 , predicted:0 
selected:0 , predicted:0
selected:1 , predicted:1
selected:1 , predicted:1
selected:1 , predicted:1
selected:1 , predicted:1
selected:1 , predicted:1
selected:2 , predicted:2
selected:2 , predicted:2
selected:2 , predicted:2
selected:2 , predicted:2
selected:2 , predicted:2
selected:3 , predicted:3
selected:3 , predicted:3
selected:3 , predicted:3
selected:3 , predicted:3
selected:3 , predicted:3
```
## graphic result 
Random generation:

![Figure_1](https://github.com/sikorskayaX/perceptron/assets/106336275/70f74bde-a9c5-4fff-a4db-33f60157e542)

One more random generation:

![Figure_2](https://github.com/sikorskayaX/perceptron/assets/106336275/9cebb63c-9b16-4c09-b394-79daaf87480a)

