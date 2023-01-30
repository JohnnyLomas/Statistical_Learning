import numpy as np

def trainNormalEquation(x: np.matrix, y: np.array) -> np.array:
	beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
	return (beta)

def classifyLinReg(beta, x):
	y = x.dot(beta)
	y = (y > 0.5).astype(int)
	return(y)

#def distance(d1: np.array, d2: np.array) -> int:
# calculate the Euclidean distance between two vectors
def euclid_dist(newpoint:np.array, trainpoint_i:np.array):
 distance = np.sqrt(sum (newpoint - trainpoint_i)**2)
 return distance

#def classifyKNN(k: int, new_data: np.matrix, data: np.matrix) -> int:
 # Find k- nearest neighbors
def distance_KNN(k: int, new_data: np.matrix, data: np.matrix):
	classified = []
	for j in range(len(new_data)):
		distances = []
		for i in range(len(data)): # for data_row in data:
			distances.append(euclid_dist(new_data[j,1:], data[i,1:]))
		
		train = data[:, 0] 
		np.append(distances, train)
		# Sort train
		# Take the smallest k values
		# Compute sum(smallest k values)/k and compare with threshold (0.5)
		# Assign prediction
		classified.append(prediction)
	
	return(classified)
 
# Make a prediction with neighbors
def predict_classification(data, new_data, k):
	neighbors = distance_KNN(k, new_data, data)
	output_values = [neighbors[:,-1]]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
# kNN Algorithm
def k_nearest_neighbors(data, new_data, k):
	predictions = list()
	for i in range(len(new_data)):
		output = predict_classification(data, new_data, k)
		predictions.append(output)
	return(predictions)




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
		return(np.append(y_vec, x_mat))

def computeError(truth, prediction):
	return(1 - sum(truth == prediction)/len(truth))

x, y = importData('zip.train', True, True)
beta = trainNormalEquation(x, y)
classified = classifyLinReg(beta, x)
error = computeError(y, classified)
print(f"\nLinear regression error (training set): {error}")

x, y = importData('zip.test', True, True)
classified = classifyLinReg(beta, x)
error = computeError(y, classified)
print(f"Linear regression error (test sest): {error}")

x_train,y_train = importData("zip.train", False)
x_test, y_test = importData("zip.test", False)
classified = k_nearest_neighbors(x_train, x_test, 1)



# Classify training by k-NN (1, 3, 5, 7, 15) -> compute error
# Classify test set by k-NN (1, 3, 5, 7, 15) -> compute error






 