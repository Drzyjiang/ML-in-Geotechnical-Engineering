import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def checkAscendingOrder(criteria:pd.DataFrame) -> bool:
    # criteria should be a 1D numpy array
    numberInterfaces = criteria.shape[0]


    for i in np.arange(1, numberInterfaces):
        if(criteria.iloc[i,0] <= criteria.iloc[i-1, 0]):
            return False
        
    return True

def getStrataIndex(criteria:pd.DataFrame, dataAggregate:pd.DataFrame)->pd.DataFrame:
    """ 
    Purpose: get index number of each data point from clustering analysis

    Format: 
    criteria -> pandas.DataFrame
    dataAggregate -> pandas.DataFrame, size = n x ?
    strataIndex: pandas.DataFrame, size = n x 1
    
    Content:
    dataAggregate first column must be Depth
    strataIndex: 0,1,... to differentiate classes

    """
    # edge case
    if criteria.empty:
        print(f"Error: criteria is empty. RandomForestStrataIndex cannot be calculated.")
        return pd.DataFrame()

    if(checkAscendingOrder(criteria) == False):
        print(f"Error: The 1D criteria array should be in ascending order")
        return pd.DataFrame()
    
    numberInterfaces = criteria.shape[0]
    numberPoints = dataAggregate.shape[0]

    strataIndex = np.zeros((numberPoints))


    # iterate over dataAggregate
    for i in np.arange(numberPoints):
        # iterate over criteria
        # check if depth is greater than the last row
        row = dataAggregate.iloc[i, :]

        if row[0] > criteria.iloc[numberInterfaces - 1, 0]:
            strataIndex[i] = numberInterfaces
        else:
            for j in np.arange(numberInterfaces):
                if row[0] < criteria.iloc[j, 0]:
                    strataIndex[i] = j    

                    break
                
    strataIndex = pd.DataFrame(strataIndex.reshape(numberPoints, 1))
    strataIndex.columns = ["Strata Index"]
    return strataIndex

            
def groupStrata(strataIndex:pd.DataFrame, dataAggregate:pd.DataFrame)->pd.DataFrame:
    '''
    Purpose: consolidate data points belonging to the same strata    

    Format:
    strataIndex: pd.DataFrame, size = n x 1
    dataAggregregate: pd.DataFrame
    dataAggregateGrouped: pd.DataFrame

    Content:
    strataIndex: strata Index for each point
    dataAggregate: aggregated data points
    dataAggregateGrouped: grouped dataAggregated
    '''

    df = pd.DataFrame(np.hstack((strataIndex, dataAggregate)))

    dataAggregateGrouped = df.groupby(0).apply(lambda x: x.to_numpy())

    return dataAggregateGrouped

def simplyStrata(dataAggregatedGrouped:pd.DataFrame, roundIntegerFlag: list[bool])->pd.DataFrame:
    '''
    Purpose:
    To average of each grouped data

    Format:
    roundIntegerFlag: list[bool]
    
    '''
    groupNumber = dataAggregatedGrouped.shape[0]

    simplifiedStrata = np.array([])
    

    for i in np.arange(groupNumber):
        # take the average value as the representative value for each layer
        simplifiedStratum = np.average(dataAggregatedGrouped[i][:, 2:], axis = 0) # skip strataIndex and depth columns

        # round to nearest integer
        simplifiedStratum = np.rint(simplifiedStratum, where = roundIntegerFlag)

        if simplifiedStrata.shape:
            simplifiedStrata = np.append(simplifiedStrata, simplifiedStratum)
        else:
            simplifiedStrata = simplifiedStratum
    

    simplifiedStrata = simplifiedStrata.reshape(groupNumber, -1)
    simplifiedStrata = pd.DataFrame(simplifiedSrata, columns = ["simplified strata"])

    return simplifiedStrata

def convertCriteria(criteria, maxDepth):
    '''
    Purpose:
    this function converts interface depths to [upper depth, lower depth] of each layer

    '''
    numberInterfaces = criteria.shape[0]
    numberLayers = numberInterfaces + 1

    criteriaConverted = np.zeros((numberLayers, 2)) # [shallower depth, deeper depth]


    for i in np.arange(numberInterfaces):
        # populate shallower depth
        criteriaConverted[i+1][0] = criteria[i]
        # populate deeper depth
        criteriaConverted[i][1] = criteria[i]

    # manually populate shallower depth of first layer as zero
    criteriaConverted[0][0] = 0

    # manually populate shallower depth of last layer as maxDepth
    criteriaConverted[numberLayers - 1][1] = maxDepth

    return criteriaConverted






