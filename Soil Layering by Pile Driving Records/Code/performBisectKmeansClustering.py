from sklearn.cluster import BisectingKMeans
from processClustering import *
from plot import *

def useBisectKmeansClustering(bisectKmeansInput, dataAggregate):
    # unzip
    performBisectKmeansFlag, numberClusters, random_state = bisectKmeansInput
    return BisectingKMeans(n_clusters = numberClusters, random_state = None).fit(dataAggregate)

def performBisectKmeans(bisectKmeansInput, dataAggregate):
    # unzip
    performBisectKmeansFlag, numberClusters, random_state = bisectKmeansInput

    if(performBisectKmeansFlag == False):
        return 0
    
    bisectKmeansClusteringObj = useBisectKmeansClustering(bisectKmeansInput, dataAggregate)

    # get results for each data
    bisectKmeansClusteringResult = processClustering(bisectKmeansClusteringObj, dataAggregate)

    # plot results for each data
    print("Below is the results by Bisecting K-Means clustering")

    labels = [ "Class", "Depth (ft)"]
    plotAggregate(bisectKmeansClusteringResult, labels, markerSize = 2)