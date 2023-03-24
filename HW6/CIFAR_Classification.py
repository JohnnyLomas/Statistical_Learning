from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data_path = '/Users/jslomas/Box/STAT_760/Statistical_Learning/HW6'

#Classes: automobile (1), cat (3), dog (5), frog (6), horse (7), ship (8), and truck (9).
classes = [1,3,5,6,7,8,9]
label_map = {1:0,3:1,5:2,6:3,7:4,8:5,9:6}

# Load cifar10 training data
cifar10 = datasets.CIFAR10(data_path, 
							train=True, 
							download=True,
							transform=transforms.ToTensor())

# Filter cifar10 to retain only the 7 desired classes
cifar7 = [(img, label_map[label]) for img, label in cifar10 if label in classes]

# Compute stats for normalization
imgs = torch.stack([img_t for img_t, _ in cifar7], dim=3)
std = imgs.view(3,-1).std(dim=1).tolist()
mean = imgs.view(3,-1).mean(dim=1).tolist()

# Reload cifar10 with normalization
cifar10 = datasets.CIFAR10(data_path, 
							train=True, 
							download=True,
							transform=transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize(mean, std)]))

# Filter cifar10 to retain only the 7 desired classes
cifar7 = [(img, label_map[label]) for img, label in cifar10 if label in classes]

# Load Cifar10 test data and filter for the 7 desired classes
cifar10_val = datasets.CIFAR10(data_path, 
							train=False, 
							download=True,
							transform=transforms.ToTensor())
cifar7_val = [(img, label_map[label]) for img, label in cifar10_val if label in classes]



len(cifar10)