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

    # dataAggregate must be a pandas.DataFrame
    scores = np.array([])
    for obj in objList:
        # training
        obj.fit(dataAggregate.iloc[:,[0]], dataAggregate.iloc[:,1:3])

        # get score for fitting
        scores = np.append(scores, obj.score(dataAggregate.iloc[:,[0]], dataAggregate.iloc[:,1:3]))

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
            if testDecisionTreeFlag == "regression":
                objList.append(tree.DecisionTreeRegressor(max_leaf_nodes = nodesVal))
            elif testDecisionTreeFlag == "classification":
                objList.append(tree.DecisionTreeClassifier(max_leaf_nodes = nodesVal))

        testLegends.append("Decision Tree")
        analyzeMaxLeafNodes(objList, nodesVals, dataAggregate, testLegends)

    # perform analysis for RandomForest
    objList = []
    if(testRandomForestFlag):
        for nodesVal in nodesVals:
            if testRandomForestFlag == "regression":
                objList.append(RandomForestRegressor(n_estimators = 10, max_leaf_nodes = nodesVal))
            elif testRandomForestFlag == "classification":
                objList.append(RandomForestClassifier(n_estimators = 10, max_leaf_nodes = nodesVal))

        testLegends.append("Random Forests")
        analyzeMaxLeafNodes(objList, nodesVals, dataAggregate, testLegends)    
 



