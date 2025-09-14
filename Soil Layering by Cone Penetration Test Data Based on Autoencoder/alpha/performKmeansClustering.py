from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from plot import *
import matplotlib.ticker as plticker


def performKmeans(kmeansClusteringInput, dataAggregate):

    '''
    Purpose:
    Top wrapper for Kmeans

    Format:
    kmeansClusteringInput: list of 
                           performKmeansFlag: bool
                           numberClusters: int
                           random_state: int
                           n_init: int
    
    dataAggregate: np.array
    
    Content:
    performKmeansFlag: whether to perform KMeans
    numberCluster: designated number of clusters
    random_state:
    n_init: number of times of k-means algorithm is run with different centroid seeds
    dataAggregate: data for clustering. First column must be depth 
    
    '''
    performKmeansFlag, numberClusters, random_state, n_init = kmeansClusteringInput

    if(performKmeansFlag == False):
        return 0
    elif( dataAggregate.size == 0 ):
        print("Error: Input data is empty.")
        return 0
    
        
    inputToUseKmeansClustering = [numberClusters, random_state, n_init]

    # split depth from feature vector
    depth = dataAggregate[:,0]
    featureVectors = dataAggregate[:,1:]

    kmeansClusteringObj = useKmeansClustering(inputToUseKmeansClustering, featureVectors)

    return kmeansClusteringObj

def useKmeansClustering(kmeansClusteringInput, featureVectors):
    '''
    Purpose: inner core of Kmeans

    Format:
    kmeansClusteringInput: list of performKmeansFlag, numberClusters, random_state, n_init
   
                           numberClusters: int
                           random_state: int
                           n_init: int
    featureVectors: np.array
    
    Content:

    numberCluster: designated number of clusters
    random_state:
    n_init: number of times of k-means algorithm is run with different centroid seeds
    featureVectors: only feature vectors. No depth
    '''
    # unzip
    numberClusters, random_state, n_init = kmeansClusteringInput

    return KMeans(n_clusters = numberClusters, random_state = random_state, n_init = n_init).fit(featureVectors)

def outputKmeansClustering(clusteringObj,  depth: np.array, plotFlag:bool = True):
    ''' 
    Purpose:
    To correlate class to each original depth point

    Format: 
    clusteringObj: sklearn Kmeans() return type
    depth: np.array, shape = (n, )
    clusteringResult: pd.DataFrame, second column is int
    plotFlag: bool

    Content:
    clusteringObj: return of sklearn Kmeans()
    depth: depth of each point
    clusteringResult: first column is depht, secon column is class label
    plotFlag: flag on whether to plot clustering results
    '''


    clusterId = clusteringObj.labels_.astype(int)

    assert clusterId.shape == depth.shape, "ERROR: data and clustering result have different size."
 
    clusteringResult = pd.DataFrame()
    clusteringResult["Depth"] = depth
    clusteringResult["Class"] = clusterId
    #clusteringResult = np.vstack((depth, clusterId)).transpose()

    if plotFlag:
        # plot results for each data
        print("Below is the results by Kmeans clustering")

        labels = ["Depth (ft)", "Class" ]
        plotAggregate(clusteringResult, labels, markerSize = 2)
        plt.show(block = False)

    return clusteringResult



def cumulativeClustering(clusteringResult):
    '''
    Purpose:
    To visualize cumulative classes count of each depth

    Format:
    clusteringResult: pd.DataFrame, shape = (n, 2)
                      first colum: float, second column: int

    Content:
    clusteringResult: first column: depth, second column: class
    
    '''

    # store all classes into a set
    classSet = set()

    for index, row in clusteringResult.iterrows():
        classSet.add(row.iloc[1].astype(int))
    
    # build a DataFrame to store count of each class
    columnList = ["Depth"]
    for classNo in classSet:
        columnList.append(str(classNo))

    cumulativeClass = pd.DataFrame(columns = columnList, dtype= 'float64')
    cumulativeClass["Depth"] = clusteringResult.iloc[:,0]

    for classNo in classSet:
        cumulativeClass[str(classNo)] = 0
    

    # iterate all depths again to fill cumulative
    for rowIndex, row in clusteringResult.iterrows():
        
        if rowIndex != 0:    
            cumulativeClass.iloc[rowIndex, 1:] = cumulativeClass.iloc[rowIndex-1, 1:]

        classCurrent = row.iloc[1].astype(int)

        colIndex = cumulativeClass.columns.get_loc(str(classCurrent))

        cumulativeClass.iloc[rowIndex, colIndex] = cumulativeClass.iloc[rowIndex, colIndex] + 1

        
    return cumulativeClass


def testKmeansK(kmeansClusteringInput: list, dataAggregate: np.array):
    '''
    Purpose:
    To determine suitable k value, using within-cluster sum of squares (WCSS) or inertia

    Format:
    kmeansClusteringInput: list of 
                           performKmeansFlag: bool
                           numberClustersList: np.array, shape = (n,)
                           random_state: int
                           n_init: int
    dataAggregate: np.array
    
    
    Content:
    kmeansClusteringInput: list of 
                           performKmeansFlag: whether to perform KMeans
                           numberClusterList: designated number of clusters
                           random_state: int
                           n_init: number of times of k-means algorithm is run with different centroid seeds
    dataAggregate: data for clustering. First column must be depth 
    '''

    # Sanity check
       
    performKmeansFlag, numberClustersList, random_state, n_init = kmeansClusteringInput


 
    # perform analysis, collect resulting objects into a list
    objList = []
    for numberClusters in numberClustersList:
        # reorganize input
        inputTemp = performKmeansFlag, numberClusters, random_state, n_init

        objList.append(performKmeans(inputTemp, dataAggregate))


    # get inertia_ from each obj
    inertiaList = []
    for obj in objList:
        inertiaList.append(obj.inertia_)
    
    axes = []
    # Plot inertia against numberClustersList
    print("Below is a plot of Within-Cluster sum of squares. Look for elbow.")
    ax = plt.plot(numberClustersList, inertiaList)
    plt.xlabel("Number of K")
    plt.ylabel("WCCS (Inertia)")
    plt.grid(True)
    #plt.yscale("log")
    xTicks = plticker.MultipleLocator(base  = 1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(xTicks)

    axes.append(ax)

    # Silhouette score
    silhouetteList = []
    for obj in objList:
        # skip k of one
        if obj.cluster_centers_.shape[0] == 1:
            silhouetteList.append(None)
        else:
            silhouetteList.append(silhouette_score(dataAggregate[:,1:], obj.labels_))

    # Plot Silhouette score against numberClustersList
    plt.figure()
    print("Below is a plot of Silhouette score Look for elbow.")
    ax = plt.plot(numberClustersList, silhouetteList)
    plt.xlabel("Number of K")
    plt.ylabel("Silhouette score")
    plt.grid(True)
    #plt.yscale("log")
    xTicks = plticker.MultipleLocator(base  = 1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(xTicks)

    axes.append(ax)

    return axes