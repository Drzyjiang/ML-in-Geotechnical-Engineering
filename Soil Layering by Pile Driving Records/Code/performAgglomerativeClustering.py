from normalization import *
from plot import *
from processClustering import *
from sklearn.cluster import AgglomerativeClustering


# Build cluster WITHOUT connectivity
def useAgglomerativeClustering(dataAggregate, numberClusters, linkageStr):

    return AgglomerativeClustering(n_clusters = numberClusters, linkage = linkageStr).fit(dataAggregate)





def performAgglomerativeClustering(agglomerativeClusteringInput, data):
    # unzip 
    performAgglomerative, normalizationAgg, linkageStr, numberClusters, referenceVector = agglomerativeClusteringInput
    
    if(performAgglomerative == False):
        return 0
    

    # copy data as dataAggregateNanAgg
    dataNormalized = None
    xlabel = ""
    ylabel = ""

    if normalizationAgg == True :
        dataNormalized = normalization(data, referenceVector)
        xlabel = "Normalized blowcounts" 
        ylabel = "Normalized depth"

        # plot normalized dataset
        plotAggregate(dataCopy, [xlabel, ylabel], markerSize = 2)
    else:
        xlabel = "Blowcounts/ft" 
        ylabel = "Depth [ft]"
    

    # perform clustering
    dataUsed = None

    if normalizationAgg == True:
        dataUsed = dataNormalized
    else:
        dataUsed = data
    
    agglomerativeClusteringObj = useAgglomerativeClustering(dataUsed, numberClusters, linkageStr)

    # get results for each data
    agglomerativeClusteringResult = processClustering(agglomerativeClusteringObj,  dataUsed)

    # plot results
    print("Below is the results by Agglomerative clustering")

    plt.figure()
    
    plotAggregate(agglomerativeClusteringResult, ["Class ID", ylabel], markerSize = 2)

    return agglomerativeClusteringObj