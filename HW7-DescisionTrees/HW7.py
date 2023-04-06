import pandas as pd
import numpy as np
import math
import random
import itertools


# Import Data 
data_pathname = "carseats.tsv"

carseats = pd.read_csv(data_pathname, sep ='\t')
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
        self.desicionFeature = None
        self.desicionSplit = None
        self.data_type = None

    def bestSplitNumeric(self, data, response):
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rss_min = np.inf
        split_min = None
        for s in quantiles:
            q = data.quantile(s)
            resp1 = response[data > q].mean()
            resp2 = response[data <= q].mean()

            rss = ((response[data > q] - resp1)**2).sum() + ((response[data <= q] - resp2)**2).sum()

            if rss < rss_min:
                rss_min = rss
                split_min = q
        
        return(rss_min, split_min)
    
    def bestSplitCategorical(self, data, response):
        options = data.unique()
        rss_min = np.inf
        split_min = None
        combinations = []
        for i in range(len(options)-1): 
            combinations = combinations + list(itertools.combinations(options, i+1))
        
        for comb in combinations:
            resp1 = response[[data[i] in comb for i in range(len(data))]].mean()
            resp2 = response[[data[i] not in comb for i in range(len(data))]].mean()

            region1 = response[[data[i] in comb for i in range(len(data))]]
            region2 = response[[data[i] not in comb for i in range(len(data))]]

            rss = ((region1 - resp1)**2).sum() + ((region2 - resp2)**2).sum()

            if rss < rss_min:
                rss_min = rss
                split_min = comb

        return rss_min,split_min

    def numericalDesicion(self, data):
        if data[self.desicionFeature] > self.desicionSplit:
            return "Go right" #TODO: Placeholder, not sure how to handle yet
        else:
            return "Go left" #TODO: Placeholder, not sure how to handle yet
    
    def categoricalDesicion(self, data):
        if data[self.desicionFeature] in self.desicionSplit:
            return "Go right" #TODO: Placeholder
        else:
            return "Go Left" #TODO: Placeholder

    def split(self, data, response, num_predictors):
        
        feature_rss_min = np.inf
        feature_min = None
        feature_split = None

        # Get a random subset of predictors
        predictors = random.sample(range(len(data.columns)), k=num_predictors)

        # Store feature names and types for future use - Categorical types must be treated differently
        features = {}
        columns = [data.columns.to_list()[i] for i in predictors]
        types = [data.dtypes.to_list()[i] for i in predictors]
        for i in range(len(data.columns)):
            features[columns[i]] = types[i]
        
        # Find the best split among the sampled predictors
        for feat in features:
            if ((features[feat] == float) | (features[feat] == int)):
                rss_min,split_min = self.bestSplitNumeric(data[feat], response)
            else:
                rss_min,split_min = self.bestSplitCategorical(data[feat], response)


            if rss_min < feature_rss_min:
                feature_rss_min = rss_min
                feature_split = split_min
                feature_min = feat
        
        # Set up decision function for the node
        # TODO: Store MSE for pruning?
        self.desicionFeature = feature_min
        self.desicionSplit = feature_split
        self.data_type = features[feature_min]
        if ((features[feature_min] == float) | (features[feature_min] == int)):
            self.decision = classmethod(self.numericalDesicion)
            rightData = data.loc[data[feature_min] > feature_split].reset_index(drop=True)
            rightResponse = response.loc[data[feature_min] > feature_split].reset_index(drop=True)
            leftData = data.loc[data[feature_min] <= feature_split].reset_index(drop=True)
            leftResponse = response.loc[data[feature_min] <= feature_split].reset_index(drop=True)
        else:
            self.decision = classmethod(self.categoricalDesicion)
            rightData = data.loc[[data.loc[i, feature_min] in feature_split for i in range(len(data))]].reset_index(drop=True)
            rightResponse = response.loc[[data.loc[i, feature_min] in feature_split for i in range(len(data))]].reset_index(drop=True)
            leftData = data.loc[[data.loc[i, feature_min] not in feature_split for i in range(len(data))]].reset_index(drop=True)
            leftResponse = response.loc[[data.loc[i, feature_min] not in feature_split for i in range(len(data))]].reset_index(drop=True)
        
        return rightData, rightResponse, leftData, leftResponse
    
    # Recursively grow the tree
    def fit(self, data, response, num_predictors, stop_number): 
        
        rightData,rightResponse,leftData,leftResponse = self.split(data, response, num_predictors)
        
        if len(rightData) > stop_number:
            rightNode = node(parent=self)
            rightNode.fit(rightData, rightResponse, num_predictors, stop_number)
            self.right = rightNode
        if len(leftData) > stop_number:
            leftNode = node(parent=self)
            leftNode.fit(leftData, leftResponse, num_predictors, stop_number)
            self.left = leftNode

dat = carseats.loc[:, "CompPrice":"US"]
resp = carseats.loc[:, "Sales"]   
mytree = node()
mytree.fit(dat, resp, 10, 10)
print("hi")
        
