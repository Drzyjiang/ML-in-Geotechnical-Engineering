import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def checkAscendingOrder(criteria):
    # criteria should be a 1D numpy array
    numberInterfaces = criteria.shape[0]

    previous = criteria[0]

    for i in np.arange(1, numberInterfaces):
        if(criteria[i] <= criteria[i-1]):
            return False
        
    return True

def getStrataIndex(criteria, dataAggregate):

    if(checkAscendingOrder(criteria) == False):
        print(f"Error: The 1D criteria array should be in ascending order")
    
    numberInterfaces = criteria.shape[0]
    numberPoints = dataAggregate.shape[0]

    strataIndex = np.zeros((numberPoints))


    # iterate over dataAggregate
    for i in np.arange(numberPoints):
        # iterate over criteria
        # check if depth is greater than the last row
        row = dataAggregate[i]

        if row[0] > criteria[numberInterfaces - 1]:
            strataIndex[i] = numberInterfaces
        else:
            for j in np.arange(numberInterfaces):
                if row[0] < criteria[j]:
                    strataIndex[i] = j    

                    break
    return strataIndex.reshape(numberPoints, 1)
            
def groupStrata(strataIndex, dataAggregate):
    # Purpose: consolidate data points belong to the same strata
    df = pd.DataFrame(np.hstack((strataIndex, dataAggregate)))

    dataAggregateGrouped = df.groupby(0).apply(lambda x: x.to_numpy())

    return dataAggregateGrouped

def simplyStrata(dataAggregatedGrouped, roundIntegerFlag):
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
    

    return simplifiedStrata.reshape(groupNumber, -1)

def convertCriteria(criteria, maxDepth):
    # this function converts criteria from form of interface to form of layer
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






