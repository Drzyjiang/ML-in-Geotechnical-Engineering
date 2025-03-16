from sklearn.ensemble import RandomForestRegressor
import numpy as np
from plot import *
import matplotlib.pyplot as plt

def useRandomForest(randomForestInput):
    # unizp input
    performRandomForestFlag, numberTrees, maxLeafNodes = randomForestInput

    return RandomForestRegressor(n_estimators = numberTrees, max_leaf_nodes = maxLeafNodes, bootstrap = True)

def getRandomForestCriteria(randomForestObj):

    criteria = []
    # iterate over all trees
    for i, eachTree in enumerate(randomForestObj):

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
                print(f"Node {node}: Feature {featureIndex}, Threshold {threshold}")

            # store threshold
            if featureIndex == 0:
                criteriaCurrent = np.append(criteriaCurrent, threshold)    
            #else:
            #    print(f"Node {node}: Leaf node")
        
        # sort criteriaCurrent
        criteriaCurrent.sort()

        # append criteriaCurrent to criteria
        #print(f"{criteriaCurrent}")
        criteria.append(criteriaCurrent)
    
    # return criteria in form of 2D numpy array
    return np.array(criteria)    



def processRandomForest(randomForestObj, dataAggregate):
    predictedValue = randomForestObj.predict(dataAggregate[:,0].reshape(-1,1))

    return np.vstack((dataAggregate[:,0], predictedValue)).transpose()


def performRandomForest(randomForestInput, dataAggregate):
    # unizp input
    performRandomForestFlag, numberTrees, maxLeafNodes = randomForestInput

    if(performRandomForestFlag == False):
        return 0
    
    # get random forest object
    randomForestObj = useRandomForest(randomForestInput)

    # fit random forest with data
    randomForestObj.fit(dataAggregate[:,0].reshape(-1,1), dataAggregate[:,1])

    # display criteria
    #getRandomForestCriteria(randomForestObj)

  
    # evaluate the score
    randomForestScore = randomForestObj.score(dataAggregate[:,0].reshape(-1,1), dataAggregate[:,1])
    print(f"The score by Random Forest: {randomForestScore}")

    return randomForestObj

def plotRandomForestResult(randomForestObj, dataAggregate):
    # get results for each data
    randomForestResult = processRandomForest(randomForestObj, dataAggregate)  

    # plot results for each data
    print("Below is the results by Random forest regression")
    labels = ["Incremental blows/foot", "Depth (ft)"]
    plotAggregate(randomForestResult, labels, markerSize = 2)



def randomForestCriteriaMedian(critiera):
    # this function reduce the random forest critieria by picking the median value
    return np.median(critiera, axis = 0)

def plotRandomForestCriteria(randomForestCriteria):
    numberTrees = randomForestCriteria.shape[0]
    numberInterfaces = randomForestCriteria.shape[1] 

    barChartX = np.arange(numberInterfaces)
    barWidth = (1- 0.2) / numberTrees

    for i in np.arange(numberTrees):
        plt.bar(barChartX + i * barWidth, randomForestCriteria[i,:],  width = barWidth, label = f"Tree {i}")
    
    plt.ylabel("Depth [ft]")
    plt.legend()
    plt.title(f"Random Forest Criteria by Each Tree")