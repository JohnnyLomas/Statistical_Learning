from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np

def readData(file):
	return pd.read_csv(file, sep=",", usecols=lambda x: x != "row.names")

train  = readData("vowel.train")
test = readData("vowel.test")

X = train.iloc[:, 1:11]
Y = train.iloc[:, 0]
model = LinearDiscriminantAnalysis()
model.fit(X, Y)

classified = []
for i in range(len(X)):
	classified.append(model.predict(X.iloc[i, 0:10].to_numpy()[:, np.newaxis].T)[0])

error = 1-sum(Y == classified)/len(X)
print(f"Training classification error rate: {error}")

X = test.iloc[:, 1:11]
Y = test.iloc[:, 0]
classified = []
for i in range(len(X)):
	classified.append(model.predict(X.iloc[i, 0:10].to_numpy()[:, np.newaxis].T)[0])

error = 1-sum(Y == classified)/len(X)
print(f"Test classification error rate: {error}")