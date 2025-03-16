# This file contain aggregate functions
import numpy as np

def aggregate_DepthVsOneFeature(dataInputToAggregate):
    aggregateFlag, data = dataInputToAggregate

    if aggregateFlag == False:
        return data

    nCol = data.shape[1]
    nDepth = data.shape[0]
    nPile = nCol -1

    depth = data.iloc[:,0]

    output = np.array([])

    # iterate over all other columns
    for i in range(1, data.shape[1]):
        #currentDepthBlows = np.hstack((depth, col))
        #print(currentDepthBlows)
        
        current = np.vstack((depth, data.iloc[:,i])).transpose()

        output = np.vstack((output, current)) if output.size else current
        #print(f"Output shape {output.shape}")

    print(f"Data is aggregated.")
    print(f"The aggregated shape is: {output.shape}")
    return output