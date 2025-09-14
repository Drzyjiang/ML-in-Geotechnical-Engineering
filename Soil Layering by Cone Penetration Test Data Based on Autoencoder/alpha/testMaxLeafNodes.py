# This function calculates scores for various max_leaf_nodes
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


def testDecisionTreeMaxLeafNodes( leafNodesRange:list[int],  dataAggregate:pd.DataFrame):
    '''
    Purpose:
    Top wrapper, to determine suitable max leaf nodes for random forest

    Format:
    leafNodesRange: list[int], len = 2
    dataAggregate: pd.DataFrame
    
    Content:
    leafNodesRange: min and max nodes range
    dataAggregatge: data 
    '''

    minLeafNodes, maxLeafNodes = leafNodesRange
    nodesVals = np.arange(minLeafNodes, maxLeafNodes)

    # perform analysis for DecisionTree
    legend = ["Decision Tree"]
    objList = []
    for nodesVal in nodesVals:
        objList.append(tree.DecisionTreeRegressor(max_leaf_nodes = nodesVal))


    analyzeMaxLeafNodes(objList, nodesVals, dataAggregate, testLegends)



def testRandomForestMaxLeafNodes(performRandomForestFlag: bool, randomForestInput: list, axes:matplotlib.axes._axes.Axes):
    '''
    Purpose:
    Top wrapper, to determine suitable max leaf nodes for random forest

    Format:
    performRandomForestFlag: bool
    randomForestInput: list of
                       profileName: str
                       dataAggregate: pd.DataFrame
                       numberTrees: int
                       testMaxLeaftNodesRange: list[int], len = 2
                       randomState: int
    axes: matplotlib.axes._axes.Axes
     
    Content:
    performRandomForestFlag: either "classification" or "regression"
    randomForestInput: list of
                       profileName: name of profile
                       dataAggregate: first column is depth, other column(s) are feature or values
                       numberTrees: number of trees in randomforest
                       testMaxLeaftNodesRange: min and max leaf nodes
                       randomState: 
    axes: axes to plot on
    '''

    # Sanity check
    assert performRandomForestFlag == "classification" or performRandomForestFlag != "regression", "ERROR: performRandomForestFlag shall be either 'classification' or 'regression'"
        
    profileName, dataAggregate, numberTrees, testMaxLeaftNodesRange, randomState = randomForestInput
    minLeafNodes, maxLeafNodes = testMaxLeaftNodesRange
    nodesVals = np.arange(minLeafNodes, maxLeafNodes)

 
    # perform analysis for RandomForest
    objList = []
    for nodesVal in nodesVals:
        if performRandomForestFlag == "regression":
            objList.append(RandomForestRegressor(n_estimators = numberTrees, max_leaf_nodes = nodesVal))
        elif performRandomForestFlag == "classification":
            objList.append(RandomForestClassifier(n_estimators = numberTrees, max_leaf_nodes = nodesVal))

  
    analyzeMaxLeafNodes(objList, nodesVals, dataAggregate, profileName, axes)    


def analyzeMaxLeafNodes(objList, nodesVals:list[int], dataAggregate:np.array, legend:str, axes:matplotlib.axes._axes.Axes = None):
    '''
    Purpose: 
    Use each trained object for prediction

    Format:
    objList: list[]
    nodesVals: list[int]
    dataAggregate: np.array
    legend: str
    notes: str
    axes: matplotlib.axes._axes.Axes

    Content:
    objList: list of trained objects
    nodesVals: list of node number
    dataAggregate: data
    legend: legend for plotting
    notes: 
    axes: axes to plot on
    '''

    scores = np.array([])

    for obj in objList:
        # training
        obj.fit(dataAggregate[:,0].reshape(-1,1), dataAggregate[:,1])

        # get score for fitting
        scores = np.append(scores, obj.score(dataAggregate[:,0].reshape(-1,1), dataAggregate[:,1]))

    # plot scores
    if axes == None:
        _, axes = plt.subplots()
    
    plotTestMaxLeafNodes(nodesVals, scores, legend, axes)
    

def plotTestMaxLeafNodes(nodesVals: float, scores:float, legend, axes = matplotlib.axes._axes.Axes):
    '''
    Purpose:
    To plot the R2 score for a give nodal value

    Format:
    nodeVal: float
    scores: float
    legends: list[str]
    axes:matplotlib.axes._axes.Axes

    Content:
    nodesVals: a list of max_leaf_nodes
    scores: scores
    legends: 
    axes: axes to plot on
    '''

    axes.plot(nodesVals, scores, marker = '.', label = legend)
    plt.xlabel("Number of max_leaf_nodes")
    plt.ylabel("R^2")
    plt.ylim((0, 1))
    plt.grid(True)
    plt.legend()
 


