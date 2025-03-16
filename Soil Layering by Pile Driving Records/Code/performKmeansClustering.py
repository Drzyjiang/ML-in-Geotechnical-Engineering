from sklearn.cluster import KMeans
from processClustering import *
from plot import *

def useKmeansClustering(kmeansClusteringInput, dataAggregate) :

    # unzip
    performKmeansFlag, numberClusters, random_state, n_init = kmeansClusteringInput

    return KMeans(n_clusters = numberClusters, random_state = random_state, n_init = n_init).fit(dataAggregate)



def performKmeans(kmeansClusteringInput, dataAggregate):

    # unzip
    performKmeansFlag, numberClusters, random_state, n_init = kmeansClusteringInput

    if(performKmeansFlag == False):
        return 0

    kmeansClusteringObj = useKmeansClustering(kmeansClusteringInput, dataAggregate)

    # get results for each data
    kmeansClusteringResult = processClustering(kmeansClusteringObj, dataAggregate)

    # plot results for each data
    print("Below is the results by Kmeans clustering")

    labels = ["Class", "Depth (ft)" ]
    plotAggregate(kmeansClusteringResult, labels, markerSize = 2)

    return 1