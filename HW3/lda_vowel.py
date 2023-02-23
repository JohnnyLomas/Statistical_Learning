import pandas as pd
import numpy as np

def readData(file):
	return pd.read_csv(file, sep=",", usecols=lambda x: x != "row.names")

train  = readData("vowel.train")

## Create mu_k, a matrix were the columns are the class-wise means for each predictor
classes = train["y"].unique()
mus = []
for cls in classes:
	cls_subset = train.loc[train["y"] == cls]
	cls_subset = cls_subset.iloc[:, 1:11]
	mus.append(cls_subset.mean(axis=0).to_numpy())
mus = np.column_stack(mus)

## Create pi_k, a vector of the classwise prior estimates
pi_k = train.groupby(["y"])["y"].count()/len(train)

## Create sigma, the covariance matrix
#sigma = np.cov(train.iloc[:, 1:11], rowvar=False)
sigma = []
for cls in classes:
	cls_subset = train.loc[train["y"] == cls]
	cls_subset = cls_subset.iloc[:, 1:11].to_numpy()
	mult = []
	for i in range(len(cls_subset)):
		mult.append((cls_subset[i, :, np.newaxis] - mus[:, cls-1, np.newaxis]).dot((cls_subset[i, :, np.newaxis] - mus[:, cls-1, np.newaxis]).T))
	sigma.append(sum(mult))

sigma = sum(sigma)/(len(train)-len(classes))

def classifyLDA(x, mu, pi, sig, classes):
	x = x[:, np.newaxis]
	
	delta = []
	for c in classes:
		mu_k = mu[:, c-1, np.newaxis]
		delta.append(float(x.T.dot(np.linalg.inv(sig)).dot(mu_k) - (mu_k.T.dot(np.linalg.inv(sig)).dot(mu_k))/2 + np.log(pi[c])))
	
	return max(range(len(delta)), key=delta.__getitem__) + 1

data = train.iloc[:, 1:11].to_numpy()
classified = []
for i in range(len(data)):
	classified.append(classifyLDA(data[i,:], mus, pi_k, sigma, classes))

error = 1-sum(train["y"] == classified)/len(train)
print(f"Training classification error rate: {error}")

# Classify the test set
test = readData("vowel.test")
data = test.iloc[:, 1:11].to_numpy()
classified = []
for i in range(len(data)):
	classified.append(classifyLDA(data[i,:], mus, pi_k, sigma, classes))

error = 1-sum(test["y"] == classified)/len(test)
print(f"Test classification error rate: {error}")