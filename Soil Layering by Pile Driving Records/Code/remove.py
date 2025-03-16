# This file contains functions to remove 
import numpy as np


def removeNan(dataInputToRemoveNan):
    # this function removes a row, if any feature is nan
    removeNanFlag, dataAggregate = dataInputToRemoveNan

    if removeNanFlag == False:
        return dataAggregate
    


    dataAggregateNoNan = np.array([])

    for row in dataAggregate:
        skip = False

        for col in row:
            if np.isnan(col):
                skip = True
                break

        # skip the row with any nan
        if not skip:
            #print (dataAggregateNoNan)
            if dataAggregateNoNan.size:
                dataAggregateNoNan = np.vstack((dataAggregateNoNan, row ))
            else:
                dataAggregateNoNan = row
    print(f"Before removeNan, the row number is {dataAggregate.shape[0]}")
    print(f"After removeNan, the row number is {dataAggregateNoNan.shape[0]}")
    return dataAggregateNoNan