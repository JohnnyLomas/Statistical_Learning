import pandas as pd
import numpy as np
import math
import random

def readData(file):
	data = pd.read_csv(file, sep=",")
	
	# Recode 'famhist' as dummy variable
	data["famhist"] = data["famhist"].replace({"Present":1, "Absent":0})

	data = np.column_stack([np.ones(len(data)) ,data.iloc[:,range(1,11)].to_numpy()])
	return data

def computeProbabilities(x, b):
	# x and b are assumed to be two column vectors
	# of equal length
	eq = b.T.dot(x)
	prob = math.exp(eq)/(1 + math.exp(eq))
	return(prob)

def fitLogistic(x_data, y_data, b_old, maxIter):
	iter = 0
	run = True
	while run == True and iter < maxIter:
	
		# Compute vetor of fitted probabilities (p)
		p = np.array([computeProbabilities(x_vec, b_old) for x_vec in x_data])

		# Compute the weights matrix (w)
		weights = p*(1-p)
		w = np.identity(len(x_data))

		# Compute the newton-raphson step
		b_new = b_old + np.linalg.inv(x_data.T.dot(w).dot(x_data)).dot(x_data.T).dot(y_data - p)
		iter += 1

		# Check convergence threshold
		run = sum(abs(b_new-b_old) >= threshold) > 0
		if not run:
			break
		else: 
			b_old = b_new

	if iter == maxIter and run == True:
		print(f"Fitting stopped after max iterations: {maxIter}...")
	#else:
	#	print(f"Fitting complete after {iter} iterations.")

	return(b_new)

data = readData("SAheart.data")

## Set initial value for parameters
param_num = np.shape(data)[1]-1
b_old = np.zeros([param_num])

## Set threshold for convergence
threshold = 0.000001
maxIter = 10000

## Set bootstrap parameters
n_iter = 100
sample_n = math.ceil(len(data)*.75)

## Perform bootstrap calculations
betas = np.zeros(shape=(n_iter, param_num))
for i in range(n_iter):
	sample = np.array(random.choices(data, k=sample_n))
	b_hat = fitLogistic(sample[:, 0:10], sample[:, 10], b_old, maxIter)
	betas[i,:] = b_hat

results = pd.DataFrame({"Parameter": ["Intercept", "sbp","tobacco","ldl","adiposity","famhist","typea","obesity","alcohol","age"],
						"Mean": np.mean(betas, axis=0),
						"Std": np.std(betas, axis = 0)})

print(f"Parameter stats with {n_iter} bootstraps:")
print(results)