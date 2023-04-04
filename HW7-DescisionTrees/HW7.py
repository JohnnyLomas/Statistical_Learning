import pandas as pd
import numpy as np
import math
import random
from sklearn.utils import shuffle


# Import Data 
data_pathname = "/Users/ntb3/Documents/STAT_760_Statistical_Learning/Statistical_Learning/HW7-DescisionTrees/carseats.tsv"

carseats = pd.read_csv("/Users/ntb3/Documents/STAT_760_Statistical_Learning/Statistical_Learning/HW7-DescisionTrees/carseats.tsv", sep ='\t')
carseats = carseats.sample(frac=1) # randomize the data 

#Data Split into 10 Subsets 
sets = []
for i in range(1, 11):
    sets += [i] * 40

carseats["sets"] = sets  

# Need Training and test 


#response = carseats.loc[:,"Sales"]
#data = carseats.loc[:, "CompPrice":]

#carseats =(carseats-carseatsmean())/carseats.std() # THis will normalize the data should we need to 

# Set up 10fold cross-validation 


class node:
    def __init__(self, parent = None, left = None, right = None):
        self.parent = parent
        self.left = left
        self.right = right


    def fit(self, data, response, complexity, prune=False):
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        feature_rss_min = np.inf
        feature_min = None
        feature_split = None
        for feat in features:
            rss_min = np.inf
            split_min = None
            for s in quantiles:

                q = data.feature.quantile(s)
                resp1 = response[data.feature > q].mean()
                resp2 = response[data.feature <= q].mean()

                rss = ((data[data.feature > q] - resp1)**2).sum() + ((data[data.feature <= q] - resp2)**2).sum()

            if rss < rss_min:
                rss_min = rss
                split_min = q

        if rss_min < feature_rss_min:
            feature_rss_min = rss_min
            feature_split = split_min
            feature_min = feat

        # Initialize CHILDREN based on split
        # If split causes less data than 'complexity' stop

class tree:
    def _init_(self):
        self.tree = []
    
    #def fit(self, data): # the thing that grows the tree


