import numpy as np

def calcNormalEquation(x: np.matrix, y: np.matrix) -> np.array:
	theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
	return (theta)

def distance(d1: np.array, d2: np.array) -> int:
	# TODO: calculate distance in 256 

def calcKNN():
	#TODO

def importData():
	# TODO: Import 2s and 3s
	# import as matrix
	# code 2 -> 0, 3 -> 1

def computeError():


def main():
	# Import training data
	# Train linear regression
	# Classify training set by linear regression -> compute error
	# Classify test set by linear regression -> compute error
	# Classify training by k-NN (1, 3, 5, 7, 15) -> compute error
	# Classify test set by k-NN (1, 3, 5, 7, 15) -> compute error
	#
