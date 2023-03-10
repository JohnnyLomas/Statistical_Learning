import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

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

    coords = [(round(random.uniform(0,1),5), round(random.uniform(0,1),5)) for _ in range(1200)]
    coords = pd.DataFrame(coords, columns = ['x', 'y'])
    
    coords['class'] = coords.apply(lambda x: sort_coordinates(x.x, x.y), axis=1)
    
    coordsA = coords[coords['class'] == 1]
    coordsB = coords[coords['class'] == 0]
    coordsA = coords[coords['class'] == 1].sample(300, replace=False, random_state=seed)
    coordsB = coords[coords['class'] == 0].sample(300, replace=False, random_state=seed)

    return pd.concat([coordsA, coordsB], axis=0)

def plotData(data, cl):
    fig, ax = plt.subplots()
    colors = {1:'red', 0:'green'}
    ax.scatter(data['x'], data['y'], c=data[cl].map(colors))
    plt.show()

class unit:
    def __init__(self, num):
        self.forward_value = 0
        self.backward_value = 0
        self.number = num
        self.delta = 0
        if num == 0:
            self.forward_value = 1
    
    def activation(self, x):
        return(1/(1+math.exp(-x)))
    
    def derivative(self, x):
        return(self.activation(x)*(1-self.activation(x)))
    
    def forward(self, x):
        self.forward_value = self.activation(x)
    
    def backward(self, x):
        self.backward_value = self.derivative(x)

    def setF(self, x):
        self.forward_value = x
    
    def setB(self, x):
        self.backward_value = x

class vanilla_network:
    # Class represents a network with two input units (plus intercept), a variable
    # number of hidden units (plus intercept unit), and a single output unit. The values
    # of y must be coded as 1 and 0.
    def __init__(self, x, y, hidden, weights, betas):
        self.input_layer = [unit(0), unit(1), unit(2)]
        self.hidden_layer = [unit(0)] + [unit(i+1) for i in range(hidden)]
        self.output = unit(1)
        self.loss = unit(1)
        self.weights = weights
        self.betas = betas
        self.x = x
        self.y = y
        self.deltas_2 = np.zeros([hidden +1, 1])
        self.deltas_1 = np.zeros([3,hidden])

        # Set value of input units
        i = 0
        for u in self.input_layer:
            if i == 0:
                i += 1
                continue
            u.setF(self.x[i-1])
            i += 1

    def feedForward(self):
        # Compute forward value of hidden units from input units
        j = 0
        for hidden in self.hidden_layer:
            if j == 0:
                j+=1
                continue
            
            val = 0
            i = 0
            for input in self.input_layer:
                val = val + input.forward_value*self.weights[i,j-1]
                i += 1
            
            hidden.forward(val)
            hidden.backward(val)
            j += 1
        
        # Compute forward value of the single output unit
        val  = 0
        i = 0
        for hidden in self.hidden_layer:
            val = val + hidden.forward_value*self.betas[i]
            i += 1
        
        # Set value of the output unit in terms of probability using the sigmoid function (same as activation)
        self.output.forward(val)
        self.output.backward(val)
        
        # Set the forward and backward values of the single error unit
        self.loss.setF(-(self.y*math.log(self.output.forward_value + 0.0000000001))-((1-self.y)*math.log(1.00000001-self.output.forward_value)))
        self.loss.setB((-((self.y-1)/(1.00000001-self.output.forward_value))-(self.y/(self.output.forward_value+0.0000000001))))
    
    def backPropagate(self):
        i = 0
        for hidden in self.hidden_layer:
            self.deltas_2[i] = self.loss.backward_value*self.output.backward_value*hidden.forward_value
            i += 1
        
        i = 0
        for hidden in self.hidden_layer:
            if i == 0:
                i+=1
                continue
            j = 0
            for input in self.input_layer:
                self.deltas_1[j,i-1] = self.loss.backward_value*self.output.backward_value*self.betas[i-1]*hidden.backward_value*input.forward_value
                j += 1
            i += 1
    
    def predict(self):
        if self.output.forward_value > 0.5:
            return 1
        else:
            return 0

# Set up the full network
data = generateData(123)
k = 16
lr  =0.01
weight = np.random.rand(3,k)-.5
beta = np.random.rand(k+1,1)-.5
full_network = [vanilla_network(np.array([x,y]), int(cl), k, weight, beta) for x,y,cl in zip(data["x"], data["y"], data["class"])]

for e in range(40):
    for net in full_network:
        net.feedForward()
        net.backPropagate()

    #Update parameters
    for i in range(len(beta)):
        beta[i] = beta[i] - lr*sum([net.deltas_2[i] for net in full_network])

    for i in range(np.shape(weight)[0]):
        for j in range(np.shape(weight)[1]):
            weight[i,j] = weight[i,j] - lr*sum([net.deltas_1[i,j] for net in full_network])

    for net in full_network:
        net.betas = beta
        net.weights = weight

    print(f"Loss: {sum([net.loss.forward_value for net in full_network])}")

prediction = [net.predict() for net in full_network]
data["prediction"] = prediction
plotData(data, "prediction")

grid = np.zeros([10000,2])
l = 0
for i in range(100):
 for j in range(100):
     grid[l] = np.array([i/100, j/100])
     l += 1

prediction = np.zeros([10000, 1])
for i in range(len(grid)):
    net = vanilla_network(grid[i], 1, k, weight, beta)
    net.feedForward()
    prediction[i] = int(net.predict())

grid = np.column_stack([grid,prediction])
grid_df = pd.DataFrame(grid, columns =['x', 'y', 'pred'])
plotData(grid_df, "pred")