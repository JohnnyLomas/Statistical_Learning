import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable

# Load the data from CSV
housing_data = pd.read_csv("/Users/ntb3/Documents/STAT_760_Statistical_Learning/CaliforniaHousing/cal_housing.data", header=None)
housing_data=(housing_data-housing_data.mean())/housing_data.std()

def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

def forward(x, w, b):
    return torch.matmul(x, w) + b
    #return w.data*x.data + b.data   


features = housing_data.iloc[:, :-1].values
target = housing_data.iloc[:, -1].values



x = torch.from_numpy(features).float() # Convert the data to PyTorch tensors
y = torch.from_numpy(target).float()

y = y.reshape(-1, 1) # Reshape the target variable to be a 2D tensor


# Define the number of input features
num_features = features.shape[1]

# Initialize the weights and biases
w = torch.randn(num_features, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Define the learning rate and number of epochs
learning_rate = 0.01
num_epochs = 1000

# Train the model
for epoch in range(num_epochs):
    # Forward pass
    y_pred = forward(x, w, b)
    loss = mse_loss(y_pred, y)
    
    # Backward pass
    loss.backward()
    
    w.data = w.data - learning_rate*w.grad.data
    b.data = b.data - learning_rate*b.grad.data
        
    # Reset gradients to zero
    w.grad.zero_()
    b.grad.zero_()
        
    # Print the loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


print(f"Weights: \n {w.data}")
print(f"Intercept: \n {b.data}")

