import pandas as pd
import numpy as np
from scipy.stats import entropy

def criteriaToClass(randomForestCriteriaReduced: pd.DataFrame, depth: pd.DataFrame):
    '''
    Purpose:
    Based on criteriaReduced, assign class to each depth

    Format:
    randomForestCriteriaReduced: pd.DataFrame, shape = n x 2
    depth: pd.DataFrame, shape = n x 1

    Content: 
    randomForestCriteriaReduced: first column: interface depth sorted, second column: class number sorted 
    depth: depth to assigned with class
    '''

    output = depth.copy()
    output["Class"] = -1

    for indexOutput, rowOutput in output.iterrows():
        currentDepth = rowOutput.iloc[0]

        # check if currentDepth is greater than the last interface
        lastInterfaceDepth = randomForestCriteriaReduced.iloc[-1, 0]

        if currentDepth > lastInterfaceDepth:
            
            output.loc[indexOutput, "Class"] = randomForestCriteriaReduced.iloc[-1,1] + 1
        else:
            # iterate over criteria
            for index, row in randomForestCriteriaReduced.iterrows():
                interfaceDepth = row.iloc[0]
                if currentDepth <= interfaceDepth:
                    output.loc[indexOutput, "Class"] = row.iloc[1]
                    break
            
    return output


def unifyClassByLayering(globalClassIds, layeringCriteria):
    '''
    Purpose: 
    Segregate globalClassIds by layeringCriteria, count the majority votes of each layer, and return the major globalClassId of each layer

    Type:
    globalClassIds: np.array, dtypes = float64, size = n x 2
    layeringCriteria: pd.DataFrame, dtypes = int64, size = m x 2

    Content:
    globalClassIds: first column is depth, second column is globalClassId
    layeringCriteria: first column is interface depth
                      ["Class"] is globalClassId
    '''

    # Sanity check of range of criteria
    depthMax = globalClassIds[:,0].max()
    depthMin = globalClassIds[:,0].min()

    criteriaMax = layeringCriteria.iloc[:,0].max()
    criteriaMin = layeringCriteria.iloc[:,0].min()

    assert criteriaMax <= depthMax and criteriaMax >= depthMin, "ERROR: The max interface depth is not in range of depth."
    assert criteriaMin <= depthMax and criteriaMin >= depthMin, "ERROR: The min interface depth is not in range of depth." 

    numInterface = layeringCriteria.shape[0]
    numLayers = numInterface + 1

    # get unique number of Ids
    uniqueIds = np.unique(globalClassIds[:,1])
    uniqueIds.sort()

    # map uniqueIds to column Indices
    mappedColumnIndices = np.arange(uniqueIds.size)

    uniqueIdsToColumnIndices  = dict(zip(uniqueIds, mappedColumnIndices))

    # use map to store count for each layer
    occurrences = np.zeros((numLayers, uniqueIds.size))

    # iterate over each row of globalClassIds
    for row in globalClassIds:
        currentDepth = row[0]
        currentClassId = row[1]
        currentLayer = numLayers - 1

        # iterate over row of criteria
        for i in np.arange(numInterface):
            interfaceDepth = layeringCriteria.iloc[i,0]

            if currentDepth <= interfaceDepth:
                currentLayer = i
                break
        
        # find mappedColumnIndex
        mappedColumnIndex = uniqueIdsToColumnIndices[currentClassId]

        # increment output[currentLayer][mappedColumnIndex]
        occurrences[currentLayer][mappedColumnIndex] = occurrences[currentLayer][mappedColumnIndex] + 1
    
    # for each row, get the column Index corresponding to most votes
    majorityVoteColumnIndices = np.argmax(occurrences, axis = 1)
    majorityPercentage = np.max(occurrences, axis = 1) / np.sum(occurrences, axis = 1)

    # map majoirtyVoteColumnIndex back to uniqueId
    majorityIds = []
    
    for majorityVoteColumnIndex in majorityVoteColumnIndices:
        classId = uniqueIdsToColumnIndices[majorityVoteColumnIndex]
        majorityIds.append(classId)

    majorityIds = np.array(majorityIds)
    layerId = np.arange(numLayers).astype(str)
    probability = (occurrences.transpose()/ occurrences.sum(axis = 1)).transpose()
    entropyValue = entropy(probability, axis = 1)

    # convert results to pd.DataFrame
    output = np.vstack((layerId, majorityIds, majorityPercentage, entropyValue)).transpose()    
    output = pd.DataFrame(output, columns = ["Layer", "Class", "Major Class Percentage", "Entropy"])

    output["Major Class Percentage"] = output["Major Class Percentage"].astype(float)
    output["Entropy"] = output["Entropy"].astype(float)

    return output

