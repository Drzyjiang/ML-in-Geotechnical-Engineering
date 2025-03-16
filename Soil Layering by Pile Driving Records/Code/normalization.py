import numpy as np

def normalization(dataAggregate, referenceVector):
    # Normalization
    # reshape referenceVector
    referenceVector = np.reshape(referenceVector, (1,-1))

    # reference vector must have the same number of columns as dataAggregateNan
    print(dataAggregate.shape)
    print(referenceVector.shape)
    assert dataAggregate.shape[1] == referenceVector.shape[1], "The feature number of data and referenceVector should be same"

    # check referenceVector has no zeros
    for i in range(referenceVector.shape[1]):
        assert referenceVector[0][i] != 0, "the reference for " + str(i) + "th feature cannot be zero"
        
    dataAggregate /= referenceVector

    return dataAggregate