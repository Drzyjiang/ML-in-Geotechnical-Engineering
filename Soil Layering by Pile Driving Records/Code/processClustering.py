import numpy as np


def processClustering(clusteringObj,  dataAggregate):
    # correlate class to each original point
    clusterId = clusteringObj.labels_

 
    clusteringResult = np.vstack((dataAggregate[:,0], clusterId)).transpose()
    

    return clusteringResult