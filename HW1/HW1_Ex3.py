import numpy

def calcNormalEquation(x: numpy.matrix, y: numpy.matrix) -> numpy.array:
	theta = numpy.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
	return (theta)

