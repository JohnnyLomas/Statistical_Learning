import pandas as pd
import numpy as np



def readData(file):
	return pd.read_csv(file, sep=",")

SAheart = readData("SAheart.data")
print(SAheart)