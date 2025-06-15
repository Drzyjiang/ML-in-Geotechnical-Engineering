from sklearn import tree
import numpy as np
import pandas as pd
from plot import *


def useDecisionTree(performDecisionTreeFlag: str, numberClusters:int, randomState):
    '''
    Purpose:
    To generate a decision tree object

    Format:
    performDecisionTreeFlag: str
    numberClusters: int
    randomState: int

    Content:
    performDecisionTreeFlag: types of analysis, ["regression", "classification"]
    numberClusters: number of clusters/layers
    randomState: random state
    '''
    
   
    if performDecisionTreeFlag == "regression":
        return tree.DecisionTreeRegressor(max_leaf_nodes = numberClusters, monotonic_cst = None, random_state = randomState)
    elif performDecisionTreeFlag == "classification":
        return tree.DecisionTreeClassifier(max_leaf_nodes = numberClusters, monotonic_cst = None, random_state = randomState)

def getDecisionTreeCriteria(decisionTreeObj) -> pd.DataFrame:
    '''
    Purpose:
    To extract splitting crteria formed from decision tree
    
    Content:
    decisionTreeObj: decisionTree Objective after fitting
    '''
    decisionTreeCriteria = np.array([])

    # Print the criteria used at each node
    for node in range(decisionTreeObj.tree_.node_count):
        # check if it's a decision node
        if decisionTreeObj.tree_.children_left[node] != decisionTreeObj.tree_.children_right[node]: 
            feature_index = decisionTreeObj.tree_.feature[node]
            threshold = decisionTreeObj.tree_.threshold[node]
            #print(f"Node {node}: Feature index {feature_index}, Threshold {threshold}") 
            
            # append threshold
            decisionTreeCriteria = np.append(decisionTreeCriteria, threshold)

    decisionTreeCriteria = np.sort(decisionTreeCriteria)

    decisionTreeCriteria = pd.DataFrame(decisionTreeCriteria, columns = ["Decision Tree Criteria"])

    return decisionTreeCriteria

def processDecisionTree(decisionTreeObj, dataAggregate:pd.DataFrame)->pd.DataFrame:
    '''
    Purpose:
    To use trainned decision tree objective to predict results from the depth data in dataAggregate

    Format:
    decistionTreeObj: ?
    dataAggregrate: pd.DataFrame, size = n * x
    output: pd.DataFrame

    Content:
    decisionTreeObj: trained decision tree objective
    dataAggregate: data, first column must be depth
    output: predicated results

    '''
    # dataAggregate is pandas.dataframe
    predictedValue = decisionTreeObj.predict(dataAggregate.iloc[:,0].to_frame())

    # convert numpy.array to pandas.dataframe
    columnNames = dataAggregate.columns.tolist()
    columnNames = columnNames[1:]

    predictedValue = pd.DataFrame(predictedValue, columns = columnNames)


    output = pd.concat([dataAggregate.iloc[:,0], predictedValue], axis = 1)

    return output


def performDecisionTree(performDecisionTreeFlag:str, decisionTreeInput):
    '''
    Purpose: top wrapper of performing decision tree

    Format:
    performDecisionTreeFlag: str
    decisionTreeInput: pandas.DataFrame, size = n * x

    Content:
    performDecisionTreeFlag = "regression", "classification"
    decisionTreeInput: first column must be depth
    '''
    # unzip
    dataAggregate, numberClusters, randomState = decisionTreeInput

    if(performDecisionTreeFlag != "regression" and performDecisionTreeFlag != "classification"):
        return 

    # get decision tree object
    decisionTreeObj = useDecisionTree(performDecisionTreeFlag, numberClusters, randomState)

    # fit decisionTreeObj with data


    # must reshape X to a 2D array, even if it only has one column
    decisionTreeObj = decisionTreeObj.fit(dataAggregate.iloc[:,[0]], dataAggregate.iloc[:,1:])

    # display criteria
    #getDecisionTreeCriteria(decisionTreeObj)    

    # get results for each data
    decisionTreeResult = processDecisionTree(decisionTreeObj, dataAggregate)

    # evaluate the score
    # Need to convert 1D array, i.e., depth to 2D
    decisionTreeScore = decisionTreeObj.score(dataAggregate.iloc[:, [0]], dataAggregate.iloc[:,1:3])
    print(f"The score by Decision tree regression: {decisionTreeScore}")

    return [decisionTreeObj, decisionTreeResult]

def plotDecisionTreeResult(decisionTreeResult: pd.DataFrame, dataAggregate: pd.DataFrame, axes:matplotlib.axes._axes.Axes = None) ->list[matplotlib.axes._axes.Axes]:
    '''
    Purpose: 
    To plot decision tree prediction results

    Format:
    decisionTreeResult: pd.DataFrame, size = n * x
    dataAggregate: pd.DataFrame

    Content:
    decisionTreeResult: decision tree result
    dataAggregate: source data
    '''

    plotNumbers = decisionTreeResult.shape[1] -1 


    if axes is None:
        fig, axes = plt.subplots(1, plotNumbers)

        # must pass a list
        if plotNumbers == 1:
            axes = [axes]



    # plot results for each data
    print("Below is the results by Decision tree regression")
    labels = dataAggregate.columns.tolist()

    axesResult = plotAggregate(decisionTreeResult, labels, markerSize = 2,  axes = axes)


    return axesResult

