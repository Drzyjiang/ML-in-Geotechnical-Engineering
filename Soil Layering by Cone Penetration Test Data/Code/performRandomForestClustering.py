from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from plot import *
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def useRandomForest(performRandomForestFlag: str, randomForestInput):
    '''
    Purpose:
    To generate random forest object

    Format:
    performRandomForestFlag: str
    randomForestInput: list

    Content:
    performRandomForestFlag: analysis type, either "classification" or "regression"
    randomForestInput:
    '''
    # unizp input
    data, numberTrees, maxLeafNodes, randomState = randomForestInput

    if performRandomForestFlag == "classification":
        print(f"Use Random Forest classifier.")
        return RandomForestClassifier(n_estimators = numberTrees, max_leaf_nodes = maxLeafNodes, bootstrap = True, random_state = randomState)
    elif performRandomForestFlag == "regression":
        print(f"Use Random Forest regressor.")
        return RandomForestRegressor(n_estimators = numberTrees, max_leaf_nodes = maxLeafNodes, bootstrap = True, random_state = randomState)

def getRandomForestCriteria(randomForestObj, printFlag = False) -> pd.DataFrame:
    '''
    Purpose: 
    To extract splitting criteria when establishing random forest

    Format:
    randomForestObj:
    pringFlag: bool
    criteria: pd.DataFrame, size = numberOfTree x (numberClusters - 1)

    Content:
    randomForestObj: trained random forest objective
    printFlag: whether to print out splitting criteria
    criteria: splitting criteria of each tree
    '''
    criteria = []
    # iterate over all trees
    for i, eachTree in enumerate(randomForestObj):

        if printFlag:
            print(f"Tree {i}:")

        # Access the tree structure
        treeStructure = eachTree.tree_

        criteriaCurrent = []

        # iterate over eachTree
        for node in range(treeStructure.node_count):
            featureIndex = treeStructure.feature[node]
            threshold = treeStructure.threshold[node]

            # check if the node is a leaf node
            if featureIndex != -2: # -2 indicates a leaf node in sklearn
                if printFlag:
                    print(f"Node {node}: Feature {featureIndex}, Threshold {threshold}")

            # store threshold
            if featureIndex == 0:
                criteriaCurrent = np.append(criteriaCurrent, threshold)    
            else:
                if printFlag:
                    print(f"Node {node}: Leaf node")
        
        # sort criteriaCurrent
        criteriaCurrent.sort()

        # append criteriaCurrent to criteria
        criteria.append(criteriaCurrent)
    
    # return criteria in form of 2D DataFrame
    return pd.DataFrame(np.array(criteria)) 



def processRandomForest(randomForestObj, dataAggregate: pd.DataFrame) ->pd.DataFrame :
    '''
    Purpose:
    To use trained random forest object for prediction

    Format:
    randomForestObj:
    dataAggregate: pd.DataFrame,

    Content:
    randomForestObj: trained random forest object
    dataAggregate: first column should be depth
    
    '''
    predictedValue = randomForestObj.predict(dataAggregate.iloc[:,[0]])

    # convert predictedValue to pandas.DataFrame
    columnNames = dataAggregate.columns.tolist()
    columnNames = columnNames[1:]

    predictedValue = pd.DataFrame(predictedValue, columns = columnNames)
    return pd.concat([dataAggregate.iloc[:,[0]], predictedValue], axis = 1)


def performRandomForest(performRandomForestFlag:str, randomForestInput):
    '''
    Purpose:
    Top wrapper for performing random forest

    Format:
    performRandomForestFlag: str
    randomForestInput: list

    Content:
    performRandomForestFlag: analysis type, either "classification", or "regression"
    randomForestInput: [dataAggregate, number of trees, max leaf nodes, random state]   
    dataAggregate: first column must be depth 
    '''
    # unizp input
    dataAggregate, numberTrees, maxLeafNodes, randomState = randomForestInput

    # dataAggregate must be a pandas.DataFrame

    if performRandomForestFlag != "classification" and performRandomForestFlag != "regression":
        return 0

    # For classification, need to ensure more unique types than numberTrees
    if performRandomForestFlag == "classification":
        numberClasses =  calculateClasses(dataAggregate)
        
        if numberClasses < maxLeafNodes:
            print("ERROR: Distinctive class number is less than number of clusters.")
            print(f"There are {maxLeafNodes} nodes, but there are only {numberClasses} unique classes.")
            return

    
    # get random forest object
    randomForestObj = useRandomForest(performRandomForestFlag, randomForestInput)

    # fit random forest with data
    # X must be a 2D array, even if it only has one column
    randomForestObj.fit(dataAggregate.iloc[:,[0]], dataAggregate.iloc[:,1:])

    # evaluate the score
    randomForestScore = randomForestObj.score(dataAggregate.iloc[:,[0]], dataAggregate.iloc[:,1:])
    print(f"The score by Random Forest: {randomForestScore}")

    # Obtain random forest results
    randomForestResult = processRandomForest(randomForestObj, dataAggregate)  

    return randomForestObj, randomForestResult

def plotRandomForestResult(randomForestResult, dataAggregate: pd.DataFrame, axes:matplotlib.axes._axes.Axes = [None]) ->matplotlib.axes._axes.Axes:
    '''
    Purpose: 
    To plot random forest results

    Format:
    randomForestResult 
    dataAggregate: list
    axes: list[matplotlib.axes._axes.Axes]

    Content:
    randomForestResult: trained random forest result
    dataAggregate: input data    
    axes: existing axes to plot on
    '''


    # plot results for each data
    print("Below is the results by Random forest regression")
    labels = dataAggregate.columns.tolist()

    numberPlots = dataAggregate.shape[1] - 1

    if axes[0] == None:
        fig, axes = plt.subplots(1, numberPlots)

        # if numberPlots is one, need to convert axes to list[axes]
        if numberPlots == 1:
            axes = [axes]

    axesResult = plotAggregate(randomForestResult, labels, markerSize = 2, axes = axes)

    return axesResult


def randomForestCriteriaMedian(randomForestCriteria: pd.DataFrame) -> pd.DataFrame:
    '''
    Purpose:
    This function reduce the random forest critieria by picking the median value

    Format:
    criteria: pd.DataFrame, size = number of trees x (number of clusters - 1)

    content: 
    criteria: resulting random forest splitting criteria
    '''
    
    # Edge case: if criteria is empty, which means random forest does not work
    if randomForestCriteria.empty:
        print("Random forest did not work. The criteria is empty.")
        return randomForestCriteria

    #return np.median(critiera, axis = 0)
    return randomForestCriteria.median(axis = 0)

def randomForestCriteriaMajority(randomForestCriteria: pd.DataFrame) -> pd.DataFrame:
    '''
    Purpose: reduces the random forest criteria by getting groupped majority votes

    Format: 
    randomForestCriteria: np.array, size = numberTrees x (numberClusters -1)

    Content:
    randomForestCriteria: splitting criteria resulting from random forest
    '''

    # Edge case: if criteria is empty, which means random forest does not work
    if randomForestCriteria.empty:
        print("Random forest did not work. The criteria is empty.")
        return randomForestCriteria

    # resize randomForestCriteria into a 1D dataframe
    randomForestCriteria1D = randomForestCriteria.values.flatten().transpose()
  
    randomForestCriteria1D = pd.DataFrame(randomForestCriteria1D, columns = ["Random Forest Criteria 1D"])
    # aggregate using agglomerative clustering
    linkageStr = 'ward'

    numberClusters = randomForestCriteria.shape[1] 

    labels = AgglomerativeClustering(n_clusters = numberClusters, linkage = linkageStr).fit_predict(randomForestCriteria1D)

    df = pd.DataFrame(randomForestCriteria1D)

    df['Labels'] = labels

    centers = df.groupby('Labels').mean()

    # sort centers
    centers = centers.sort_values(by = "Random Forest Criteria 1D")
    return centers["Random Forest Criteria 1D"].to_frame()

def plotRandomForestCriteria(randomForestCriteria):
    '''
    Purpose: plot random forest criteria

    Format:
    randomForestCriteria: pd.DataFrame, size = numberTree x numberCluster

    Content:
    randomForestCriteria: results by random forest
    
    '''

    numberTrees = randomForestCriteria.shape[0]
    numberInterfaces = randomForestCriteria.shape[1] 

    barChartX = np.arange(numberInterfaces)
    barWidth = (1- 0.2) / numberTrees

    for i in np.arange(numberTrees):
        plt.bar(barChartX + i * barWidth, randomForestCriteria.iloc[i,:],  width = barWidth, label = f"Tree {i}")
    
    plt.ylabel("Depth [ft]")
    plt.legend(loc = "upper left", fontsize = "x-small")
    plt.title(f"Random Forest Criteria by Each Tree")

def calculateClasses(dataAggregate: pd.DataFrame)->int:
    '''
    Purpose:
    To calculate distintive number of classes

    Format:
    dataAggregate: pd.DataFrame, size = n * x, x >1

    Content:
    dataAggregate: first col shall be depth
    ''' 

    classes = np.unique(dataAggregate.iloc[:,1])
 
    return len(classes)