from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the model
class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
		self.fc1 = nn.Linear(512, 32)
		self.fc2 = nn.Linear(32, 7)

	def forward(self, x):
		out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
		out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
		out = out.view(-1, 512)
		out = torch.tanh(self.fc1(out))
		out = self.fc2(out)
		return out

model = Net().to(device)
model.load_state_dict(torch.load(data_path
                                        + '/cifar7.pt',
                                        map_location=device))
print(model)


def train():
	train_loader = torch.utils.data.DataLoader(cifar7, batch_size=64, shuffle=True)

	learning_rate = 1e-3
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	loss_fn = nn.CrossEntropyLoss()
	n_epochs = 1000
 
	for epoch in range(1, n_epochs + 1):
		loss_trn = 0.0
		for imgs, labels in train_loader:
			imgs, labels = imgs.to(device), labels.to(device)
			outputs = model(imgs)
			loss = loss_fn(outputs, labels)
 
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_trn += loss.item()
  
		if epoch == 1 or epoch % 10 == 0:
			print(f'Epoch {epoch}, Training loss {loss_trn/len(train_loader)}')

def predict():
	val_loader = torch.utils.data.DataLoader(cifar7_val, batch_size=64, shuffle=False)

	pred_stats = {
    	0:{"total":0, "correct":0, "class":"automobile"},
    	1:{"total":0, "correct":0, "class":"cat"},
    	2:{"total":0, "correct":0, "class":"dog"},
    	3:{"total":0, "correct":0, "class":"frog"},
    	4:{"total":0, "correct":0, "class":"horse"},
    	5:{"total":0, "correct":0, "class":"ship"},
    	6:{"total":0, "correct":0, "class":"truck"}
	}

	total = 0
	correct = 0

	with torch.no_grad():
		for imgs, labels in val_loader:
			imgs, labels = imgs.to(device), labels.to(device)
			batch_size = imgs.shape[0]
			outputs = model(imgs)
			_, predicted = torch.max(outputs, dim=1)
			total += labels.shape[0]
			correct += int((predicted == labels).sum())
			for i in range(len(labels.tolist())):
				pred_stats[labels[i].item()]["total"] += 1
				if predicted[i].item() == labels[i].item():
					pred_stats[labels[i].item()]["correct"] += 1

	print(f"Total Accuracy: {correct / total}")
	for cls in pred_stats.keys():
		print(f"Class Accuracy [{pred_stats[cls]['class']}]: {pred_stats[cls]['correct']/pred_stats[cls]['total']}")

predict()