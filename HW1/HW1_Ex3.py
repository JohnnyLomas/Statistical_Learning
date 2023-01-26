import numpy as np

def trainNormalEquation(x: np.matrix, y: np.matrix) -> np.array:
	theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
	return (theta)

#def classifyLinReg():

#def distance(d1: np.array, d2: np.array) -> int:
	# TODO: calculate distance in 256 

#def classifyKNN(k: int, new_data: np.matrix, data: np.matrix) -> int:
	#TODO

def importData(file: str, addOnes: bool) -> np.matrix | np.array:
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
	
	return(x_mat, y_vec)

#def computeError():


x, y = importData('zip.test', False)
# Import training data
# Train linear regression
# Classify training set by linear regression -> compute error
# Classify test set by linear regression -> compute error
# Classify training by k-NN (1, 3, 5, 7, 15) -> compute error
# Classify test set by k-NN (1, 3, 5, 7, 15) -> compute error
