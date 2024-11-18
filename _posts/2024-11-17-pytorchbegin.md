---
title: My first steps with Pytorch
tags: [python, pytorch, deep learning]
---


First, we need to import the necessary libraries.
We need to do two things:
1. Define the model
2. Create a dataset 

## The Model
The model will be a simple feedforward neural network with one hidden layer.  We will output a log softmax to get a probability distribution over the classes.  Taking the argmax of the probability distribution will give us our predicted class.

## The Dataset
We will use the iris dataset.  It is a classic dataset that is often used to test machine learning algorithms.  It has three classes and four features.  We will use the sepal length, sepal width, petal length, and petal width to predict the class of the iris flower.  Notice that that we return the x values as well as the y values as a tupple in the `__getitem__` method.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

class MultiClassNet(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_SIZE):
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_SIZE)
        self.lin2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x
    
class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
import seaborn as sns

iris = sns.load_dataset('iris')

X = torch.tensor(iris.drop('species', axis=1).values, dtype=torch.float32)
y = torch.tensor(iris['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values, dtype=torch.long)

iris_dataset = IrisDataset(X, y)

from torch.utils.data import DataLoader
train_loader = DataLoader(iris_dataset, batch_size=9, shuffle=True)

```

## Training the model
Now we can train the model.  Which is done by following these steps:
1. Define the loss function
2. Define the optimizer
3. Loop over the data in the dataloader
4. Zero the gradients (This is done before each iteration to prevent gradient accumulation)
5. Forward pass (This passes through the model to get the predictions)
6. Backward pass (This calculates the loss)
7. Step the optimizer (This updates the model parameters based on the loss)

```python

NUM_FEATURES = iris.shape[1] - 1
HIDDEN_SIZE = 10
NUM_CLASSES = 3

model = MultiClassNet(NUM_FEATURES, NUM_CLASSES, HIDDEN_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

NUM_EPOCHS = 100
losses = []
for epoch in range(NUM_EPOCHS):
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        print(y_pred)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()  
        losses.append(loss.item())

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()


```


We can see that the loss is decreasing over time.  This is a good sign that the model is learning.
We can also test the data to see how well the model is doing.

```python

from sklearn.metrics import accuracy_score

with torch.no_grad():
    y_pred = model(X)
    y_test_pred = torch.argmax(y_pred.data, dim=1)
    print(y_test_pred)
    print(accuracy_score(y_test_pred, y))


```

We can see that the accuracy is about 90%. Better than random guessing (which woud be about 40% based on the dataset!