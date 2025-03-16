# This function calculates scores for various max_leaf_nodes
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

def plotTestMaxLeafNodes(nodesVals, scores, legends):
    # nodesVals: a list of max_leaf_nodes
    # scores: scores

    plt.plot(nodesVals, scores, marker = '.')
    plt.xlabel("Number of max_leaf_nodes")
    plt.ylabel("R^2")
    plt.ylim((0, 1))
    plt.grid(True)
    plt.legend(legends)

def analyzeMaxLeafNodes(objList, nodesVals, dataAggregate, testLegends):

    scores = np.array([])

    for obj in objList:
        # training
        obj.fit(dataAggregate[:,0].reshape(-1,1), dataAggregate[:,1])

        # get score for fitting
        scores = np.append(scores, obj.score(dataAggregate[:,0].reshape(-1,1), dataAggregate[:,1]))

    # plot scores
    plotTestMaxLeafNodes(nodesVals, scores, testLegends)

def testMaxLeafNodes( leafNodesRange, testObjFlags, dataAggregate):
    testDecisionTreeFlag, testRandomForestFlag = testObjFlags

    minLeafNodes, maxLeafNodes = leafNodesRange
    nodesVals = np.arange(minLeafNodes, maxLeafNodes)

    testLegends = []
    # perform analysis for DecisionTree
    objList = []
    if(testDecisionTreeFlag):
        for nodesVal in nodesVals:
            objList.append(tree.DecisionTreeRegressor(max_leaf_nodes = nodesVal))

        testLegends.append("Decision Tree")
        analyzeMaxLeafNodes(objList, nodesVals, dataAggregate, testLegends)

    # perform analysis for RandomForest
    objList = []
    if(testRandomForestFlag):
        for nodesVal in nodesVals:
            objList.append(RandomForestRegressor(n_estimators = 10, max_leaf_nodes = nodesVal))

        testLegends.append("Random Forests")
        analyzeMaxLeafNodes(objList, nodesVals, dataAggregate, testLegends)    
 



