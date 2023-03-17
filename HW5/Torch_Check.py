import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
from torch import nn
import torch.optim as optim
import tqdm

def sort_coordinates(x,y):
    pi = math.pi
    sqrt = math.sqrt
    if sqrt((x-(1-sqrt(0.3/pi)))**2+(y-(1-sqrt(0.3/pi)))**2) <= sqrt(0.3/pi): # 30% area circle
        return 0
    elif sqrt((x-sqrt(0.2/pi))**2+(y-sqrt(0.2/pi))**2) <= sqrt(0.2/pi): 
        return 0
    elif sqrt((x-(1-sqrt(0.1/pi)))**2+(y-sqrt(0.1/pi))**2) <= sqrt(0.1/pi): # 10% area circle
        return 0
    else:
        return 1

def generateData(seed: int):
    # Class A = 1, Class B = 0
    random.seed(seed)

    coords = [(1, round(random.uniform(0,1),5), round(random.uniform(0,1),5)) for _ in range(1200)]
    coords = pd.DataFrame(coords, columns = ['intercept','x', 'y'])
    
    coords['class'] = coords.apply(lambda x: sort_coordinates(x.x, x.y), axis=1)
    
    coordsA = coords[coords['class'] == 1]
    coordsB = coords[coords['class'] == 0]
    coordsA = coords[coords['class'] == 1].sample(300, replace=False, random_state=seed)
    coordsB = coords[coords['class'] == 0].sample(300, replace=False, random_state=seed)

    return pd.concat([coordsA, coordsB], axis=0)

def toTensor(dat):
    X = torch.tensor(dat.loc[:, ("intercept", "x", "y")].values, dtype=torch.float32)
    Y = torch.tensor(dat.loc[:, "class"], dtype=torch.float32).reshape(-1, 1)
    return X,Y

def plotData(data, cl):
    fig, ax = plt.subplots()
    colors = {1:'red', 0:'green'}
    ax.scatter(data['x'], data['y'], c=data[cl].map(colors))
    plt.show()

class vanilla_network(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.hidden = nn.Linear(2, k)
        self.output = nn.Linear(k, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

k = 32
seq_model = nn.Sequential(
    nn.Linear(3,32),
    nn.Sigmoid(),
    nn.Linear(32,1),
    nn.Sigmoid()
)

def model_train(model, X_train, y_train, lr):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.SGD(model.parameters(), lr=lr)
 
    n_epochs = 1000   # number of epochs to run
 
    for epoch in range(n_epochs):
        model.zero_grad()
        output = model(X_train)
        loss =loss_fn(output,y_train)
        print(f"Loss: {loss}")
        loss.backward()
        optimizer.step()

data = generateData(123)
x,y = toTensor(data)
network  = vanilla_network(5)

model_train(seq_model, x, y, 0.1)
prediction = [seq_model(x[i]) for i in range(len(x))]
data["prediction"] = [1 if prediction[i].item() > 0.5 else 0 for i in range(len(prediction))]
plotData(data, "prediction")