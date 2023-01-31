import numpy as np

def trainNormalEquation(x: np.matrix, y: np.array) -> np.array:
	beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
	return (beta)

def classifyLinReg(beta, x):
	y = x.dot(beta)
	y = (y > 0.5).astype(int)
	return(y)

def euclid_dist(newpoint:np.array, trainpoint_i:np.array):
	distance = np.sqrt(sum((newpoint - trainpoint_i)**2))
	return distance

def classifyKNN(k: int, new_data: np.matrix, data: np.matrix):
	classified = []
	for j in range(len(new_data)):
		distances = []
		for i in range(len(data)): # for data_row in data:
			distances.append(euclid_dist(new_data[j,1:], data[i,1:]))
		
		# Make copy of responses that we can manipulate without destroying original.
		# Then associate the computed distances
		train = data[:, 0]
		distances = np.column_stack((distances, train))
		
		# Create structured array (easier to sort)
		dt = {'names':['dist', 'response'], 'formats':[float, int]}
		sort = np.zeros(len(train), dtype=dt)
		sort['dist'] = distances[:, 0]
		sort['response'] = distances[:, 1]
		
		# Sort the neighbors by distance
		sort = np.sort(sort, order='dist')
		
		# Vote on the smallest k values
		prediction = sum(sort[0:k]['response'])/k
		prediction = (prediction > 0.5).astype(int)
		
		# Assign prediction
		classified.append(prediction)
	
	return(classified)

def importData(file: str, addOnes: bool, splitResponse: bool):
	# Import data to a matrix
	matrix = np.loadtxt(file)
	
	# get only 2s and 3s
	mask = np.logical_or(matrix[:, 0] == 2, matrix[:, 0] == 3)
	matrix = matrix[mask]
	
	# code 2 -> 0, 3 -> 1
	y_vec = matrix[:, 0] == 3
	y_vec = y_vec.astype(int)

	if addOnes:
		matrix[:, 0] = np.ones(len(matrix))
		x_mat = matrix[:, 0:256]
	else:
		x_mat = matrix[:, 1:256]
	
	if splitResponse:
		return(x_mat, y_vec)
	else:
		return(np.column_stack((y_vec,x_mat)))

def computeError(truth, prediction):
	return(1 - sum(truth == prediction)/len(truth))

# Train linear regression and classify training set
x, y = importData('zip.train', True, True)
beta = trainNormalEquation(x, y)
classified = classifyLinReg(beta, x)
error = computeError(y, classified)
print(f"\nLinear regression error (training set): {error}")

# Classify test set with linear regression
x, y = importData('zip.test', True, True)
classified = classifyLinReg(beta, x)
error = computeError(y, classified)
print(f"Linear regression error (test set): {error}")

# Classify test set by KNN
train = importData("zip.train", False, False)
test = importData("zip.test", False, False)

k_list = [1,3,5,7,15]
for k in k_list:
	classified = classifyKNN(k, test, train)
	error = computeError(test[:, 0], classified)
	print(f"K-Nearest Neighbors error (k = {k}): {error}")