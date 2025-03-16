from sklearn import tree
import numpy as np
from plot import *

def useDecisionTree(decisionTreeInput, dataAggregate):
    # unzip
    performDecisionTreeFlag, numberClusters = decisionTreeInput

    #monotonicCstArray = np.array([1])
    return tree.DecisionTreeRegressor(max_leaf_nodes = numberClusters, monotonic_cst = None)

def getDecisionTreeCriteria(decisionTreeObj):
    decisionTreeCritiera = np.array([])

    # Print the criteria used at each node
    for node in range(decisionTreeObj.tree_.node_count):
        # check if it's a decision node
        if decisionTreeObj.tree_.children_left[node] != decisionTreeObj.tree_.children_right[node]: 
            feature_index = decisionTreeObj.tree_.feature[node]
            threshold = decisionTreeObj.tree_.threshold[node]
            print(f"Node {node}: Feature index {feature_index}, Threshold {threshold}") 
            
            # append threshold
            decisionTreeCritiera = np.append(decisionTreeCritiera, threshold)

    decisionTreeCritiera = np.sort(decisionTreeCritiera)

    return decisionTreeCritiera

def processDecisionTree(decisionTreeObj, dataAggregate):
    predictedValue = decisionTreeObj.predict(dataAggregate[:,0].reshape(-1,1))

    return np.vstack((dataAggregate[:,0], predictedValue)).transpose()

def performDecisionTree(decisionTreeInput, dataAggregate):
    # unzip
    performDecisionTreeFlag, numberClusters = decisionTreeInput

    if(performDecisionTreeFlag == False):
        return 

    # get decision tree object
    decisionTreeObj = useDecisionTree(decisionTreeInput, dataAggregate)

    # fit decisionTreeObj with data
    decisionTreeObj = decisionTreeObj.fit(dataAggregate[:,0].reshape(-1,1), dataAggregate[:,1])

    # display criteria
    getDecisionTreeCriteria(decisionTreeObj)    

    # get results for each data
    decisionTreeResult = processDecisionTree(decisionTreeObj, dataAggregate)

    # evaluate the score
    decisionTreeScore = decisionTreeObj.score(dataAggregate[:,0].reshape(-1,1), dataAggregate[:,1])
    print(f"The score by Decision tree regression: {decisionTreeScore}")


    # plot results for each data
    print("Below is the results by Decision tree regression")
    labels = [ "Incremental blows/foot", "Depth (ft)"]
    plotAggregate(decisionTreeResult, labels, markerSize = 2)

    return decisionTreeObj

