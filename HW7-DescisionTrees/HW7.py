import pandas as pd
import numpy as np
import math
import random
import itertools
import pydot
import uuid
import copy

######################################################################################

class node:
    def __init__(self, parent = None, left = None, right = None, graph = None):
        self.parent = parent
        self.left = left
        self.right = right
        self.desicionFeature = None
        self.desicionSplit = None
        self.data_type = None
        self.graph = graph
        self.label = str(uuid.uuid4())
        self.prediction = None
        self.cost_complexity = 0

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
            return "right" #TODO: Placeholder, not sure how to handle yet
        else:
            return "left" #TODO: Placeholder, not sure how to handle yet
    
    def categoricalDesicion(self, data):
        if data[self.desicionFeature] in self.desicionSplit:
            return "right" #TODO: Placeholder
        else:
            return "left" #TODO: Placeholder
    
    def addToGraph(self, cur_label, parent_label, data_type=None, feature=None, split=None, data_num=None, terminal=False, name=""):
        if terminal:
            lab = f"{str(split)}\n{name}"
        else:
            if ((data_type == int) | (data_type == float)):
                lab = f"{feature} > {split}\n{name}"
            else:
                lab = f"{feature} in {split}\n{name}"
        tree_graph.add_node(pydot.Node(cur_label, shape="circle", label=lab))
        tree_graph.add_edge(pydot.Edge(parent_label, cur_label, color="blue"))

    def split(self, data, response, num_predictors):
        
        self.avgResponse = response.mean()

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
            self.decision = self.numericalDesicion
            rightData = data.loc[data[feature_min] > feature_split].reset_index(drop=True)
            rightResponse = response.loc[data[feature_min] > feature_split].reset_index(drop=True)
            leftData = data.loc[data[feature_min] <= feature_split].reset_index(drop=True)
            leftResponse = response.loc[data[feature_min] <= feature_split].reset_index(drop=True)
        else:
            self.decision = self.categoricalDesicion
            rightData = data.loc[[data.loc[i, feature_min] in feature_split for i in range(len(data))]].reset_index(drop=True)
            rightResponse = response.loc[[data.loc[i, feature_min] in feature_split for i in range(len(data))]].reset_index(drop=True)
            leftData = data.loc[[data.loc[i, feature_min] not in feature_split for i in range(len(data))]].reset_index(drop=True)
            leftResponse = response.loc[[data.loc[i, feature_min] not in feature_split for i in range(len(data))]].reset_index(drop=True)
        
        self.cost_complexity = sum((rightResponse-rightResponse.mean())**2) + sum((leftResponse-leftResponse.mean())**2)

        return rightData, rightResponse, leftData, leftResponse
    
    # Recursively grow the tree
    def fit(self, data, response, num_predictors, stop_number, graph=False): 
        
        rightData,rightResponse,leftData,leftResponse = self.split(data, response, num_predictors)
        
        rightNode = node(parent=self)
        leftNode = node(parent=self)

        if len(rightData) > stop_number:
            rightNode.fit(rightData, rightResponse, num_predictors, stop_number, graph)
            if graph:
                self.addToGraph(rightNode.label, self.label, data_type=rightNode.data_type, feature=rightNode.desicionFeature, split=rightNode.desicionSplit, name=rightNode.label)
        else:
            rightNode.prediction = rightResponse.mean()
            rightNode.cost_complexity = sum((rightResponse-rightResponse.mean())**2)
            if graph:
                self.addToGraph(rightNode.label, self.label, terminal=True, split = rightNode.prediction)

        if len(leftData) > stop_number:
            leftNode.fit(leftData, leftResponse, num_predictors, stop_number, graph)
            if graph:
                self.addToGraph(leftNode.label, self.label, data_type=leftNode.data_type, feature=leftNode.desicionFeature, split=leftNode.desicionSplit, name=leftNode.label)
        else:
            leftNode.prediction = leftResponse.mean()
            rightNode.cost_complexity = sum((leftResponse-leftResponse.mean())**2)
            if graph:
                self.addToGraph(leftNode.label, self.label, terminal=True, split = leftNode.prediction)
        
        self.right = rightNode
        self.left = leftNode

    # Predict a single data point
    def predict(self, data):
        if self.prediction != None:
            return self.prediction
        
        direction = self.decision(data)
        if direction == "right":
            return self.right.predict(data)
        else:
            return self.left.predict(data)
    
    # Traverses the tree and builds a dictionary of paths
    # to candidate pruning nodes. i.e. {complexity_change:["left", "right",...]}
    def findPruningNodes(self, paths_dict, path):
        if self.right.data_type != None:
            newpath = path + ["right"]
            paths_dict = self.right.findPruningNodes(paths_dict, newpath)
        if self.left.data_type != None:
            newpath = path + ["left"]
            paths_dict = self.left.findPruningNodes(paths_dict, newpath)
        
        if (self.left.data_type == None) and (self.right.data_type == None):
            paths_dict[(self.parent.cost_complexity - self.cost_complexity)] = path
            
        return paths_dict
    
    def pruneSmallestChange(self, path):
        #candidate_nodes = self.findPruningNodes({}, [])
        #path_to_prune = candidate_nodes[min(candidate_nodes.keys())]
        if path != []:
            direction = path.pop(0)
            if direction == "right":
                self.right.pruneSmallestChange(path)
            else:
                self.left.pruneSmallestChange(path)
        else:
            self.right = None
            self.left = None
            self.data_type = None
            self.prediction = self.avgResponse
    
    def precalc_costComplexity(self):
        if self.right.prediction != None:
            rightCount,rightCC = 1, self.right.cost_complexity
        else:
            rightCount,rightCC = self.right.precalc_costComplexity()
        
        if self.left.prediction != None:
            leftCount, leftCC = 1, self.left.cost_complexity
        else:
            leftCount, leftCC = self.left.precalc_costComplexity()
        
        return (leftCount + rightCount), (leftCC + rightCC)

###################################################################################

def pruneTrees(tree: node):
    # Stops with a three terminal node tree?
    
    tree.terminal_nodes, tree.totalCostComplexity = tree.precalc_costComplexity()
    pruned_trees = [tree]
    next_tree = copy.deepcopy(tree)
    while (next_tree.left.data_type != None) | (next_tree.left.data_type != None):
        candidate_nodes = next_tree.findPruningNodes({}, [])
        path_to_prune = candidate_nodes[min(candidate_nodes.keys())]
        next_tree.pruneSmallestChange(path_to_prune)
        next_tree.terminal_nodes, next_tree.totalCostComplexity = next_tree.precalc_costComplexity()
        pruned_trees.append(copy.deepcopy(next_tree))
    return pruned_trees

def getTreeByAlpha(trees: list, alpha: list):
    cost_complexity = []
    for tree in trees:
        cost_complexity.append(tree.totalCostComplexity + alpha*tree.terminal_nodes)

    min_cost_complexity = trees[cost_complexity.index(min(cost_complexity))]

    return min_cost_complexity

def fitBestTree(data, response, alpha):
    tree = node()                           # Initialize tree
    tree.fit(data, response, 10, 10)        # Fit tree
    pruned = pruneTrees(tree)               # Prune tree to produce sequence
    bestTree = getTreeByAlpha(pruned, alpha)# Get the minimum cost complexity tree by alpha value
    return bestTree

def calcMSE(data, response, tree):
    squared_errors = []
    for i in range(len(data)):
        squared_errors.append((tree.predict(data.iloc[i,:]) - response.iloc[i])**2)
    mse = sum(squared_errors)
    return mse

# Returns the average test MSE from 10-fold cross validation
# associated with a list of alpha values (list)
def crossValidatePrunedTrees(data, response, alphas):
    # Data must be pre-randomized and have 'sets'
    # Response must also have 'sets'
    mean_mse = []
    for alpha in alphas:
        mse = []
        for set in range(1,11):
            # Get training and test subsets
            train_dat = data.loc[data["sets"] != set, data.columns != "sets"].reset_index(drop=True)
            train_resp = response.loc[response["sets"] != set, response.columns != "sets"].reset_index(drop=True).squeeze()
            test_dat = data.loc[data["sets"] == set, data.columns != "sets"].reset_index(drop=True)
            test_resp = response.loc[response["sets"] == set, response.columns != "sets"].reset_index(drop=True).squeeze()

            # Get best try by alpha value
            bestTree = fitBestTree(train_dat, train_resp, alpha)

            # Compute test MSE for this subset and alpha
            _mse = calcMSE(test_dat, test_resp, bestTree)
            mse.append(_mse)
        
        mean_mse.append(np.array(mse).mean())
    
    return mean_mse

########################################################################################

# Exercise 1a - Split data into training and test set
carseats = pd.read_csv("carseats.tsv", sep ='\t')
carseats = carseats.sample(frac=1, random_state=123)
train_dat = carseats.loc[0:350, "CompPrice":"US"].reset_index(drop=True)
train_resp = carseats.loc[0:350, "Sales"].reset_index(drop=True)
test_dat = carseats.loc[351:, "CompPrice":"US"].reset_index(drop=True)
test_resp =  carseats.loc[351:, "Sales"].reset_index(drop=True)

# Exercise 1b - Fit the tree. Analyze test performance
## Initialize tree
mytree = node()

## Initialize visualization
mytree.split(train_dat, train_resp, 10)
if ((mytree.data_type == int) | (mytree.data_type == float)):
    lab = f"{mytree.desicionFeature} > {mytree.desicionSplit}"
else:
    lab = f"{mytree.desicionFeature} in {mytree.desicionSplit}"
tree_graph = pydot.Dot("Sales Regression Tree", graph_type="graph", bgcolor="white")
tree_graph.add_node(pydot.Node(mytree.label, shape="circle", label=lab))

## Fit tree, calculate mse, and save graph
mytree.fit(train_dat, train_resp, 10, 10, graph=True)
mse = calcMSE(test_dat, test_resp, mytree)
tree_graph.write_png("Regression_Tree.png")

# Excercise 1c - Prune with cross validation and compare

# Data Split into 10 Subsets and get features and responses 
sets = []
for i in range(1, 11):
    sets += [i] * 40
carseats["sets"] = sets
data = carseats.loc[:, "CompPrice":"sets"]
response = carseats.loc[:, ("Sales", "sets")]

# Find the optimal level of complexity by pruning and 10-fold CV
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.2, 2]
mse = crossValidatePrunedTrees(data, response, alphas)
bestAlpha = alphas[mse.index(min(mse))]

# Use the best alpha to train, prune, and test performance
pruned_tree = fitBestTree(train_dat, train_resp, bestAlpha)
best_mse = calcMSE(test_dat, test_resp, pruned_tree)
print(best_mse)

# Exercise 1d - Bagging
# Bootstrap data -sample with replacment 100 sets?
    # [list of dataframes] 
# Grow 100 trees no pruning on each dataset
    # intiatilze a tree: tree = node()
    # Fit to data tree.fit(data, response, 10, 10)
    # Store the trees in a list [] - mylist = [], for loop, mylist.append(element)
# Predict on 100 trees and average response
    # tree.predict(data)

# Exercise 1e -





# Get data and response
dat = carseats.loc[:, "CompPrice":"US"]
resp = carseats.loc[:, "Sales"]  

# Initialize the tree
mytree = node()

# Initialize the tree visualization
mytree.split(dat, resp, 10)
if ((mytree.data_type == int) | (mytree.data_type == float)):
    lab = f"{mytree.desicionFeature} > {mytree.desicionSplit}"
else:
    lab = f"{mytree.desicionFeature} in {mytree.desicionSplit}"
tree_graph = pydot.Dot("Sales Regression Tree", graph_type="graph", bgcolor="white")
tree_graph.add_node(pydot.Node(mytree.label, shape="circle", label=lab))

# Fit the tree to the data using all predictors
mytree.fit(dat, resp, 10, 10)

findnodes = mytree.findPruningNodes({}, [])
mytree.predict(dat.iloc[0, :])
tree_graph.write_png("Regression_Tree.png")
print("hi")
        
#Calculate the accuracy of the tree 
def accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for i in range(total):
        if y_true[i] == y_pred[i]:
            correct += 1
    accuracy = correct / total
    return accuracy