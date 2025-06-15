import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mping
import matplotlib

def plotRawCPT(data:pd.DataFrame, fig, axes:list[matplotlib.axes._axes.Axes]):
    '''
    Purpose:
    this function plots raw CPT data
    
    Format:
    data is pandas.DataFrame, size = n x 4
    axes: list[matplotlib.axes._axes.Axes], len = 2

    Content:
    data: first column: Depth; 
          second column: Cone resistance; 
          third column: sleeve friction;
          Optional Four column: pore pressure
    axes: existing axes to plot on
    '''
    
    assert data.shape[1] >=3, "Must have at least three columns"
    
    # extract headers
    dataLabels = data.columns.tolist()

    # plot tip resistance
    invertYFlag = False

    if axes[0] == None:
        fig, axes = plt.subplots(1,2, figsize = (10,6))
        invertYFlag = True
    
    axes[0].plot(data[dataLabels[1]], data[dataLabels[0]], linestyle = '', marker = '.', markersize = 2)
    axes[0].set_xlabel(dataLabels[1])
    axes[0].set_ylabel(dataLabels[0])
    axes[0].grid(True)


    axes[1].plot(data[dataLabels[2]], data[dataLabels[0]], linestyle = '', marker = '.', markersize = 2)
    axes[1].set_xlabel(dataLabels[2])
    axes[1].set_ylabel(dataLabels[0])
    axes[1].grid(True)

    if invertYFlag:
        axes[0].invert_yaxis()
        axes[1].invert_yaxis()

    return fig, axes

def plotRfFr(depth:pd.DataFrame, Rf:pd.DataFrame, Fr:pd.DataFrame):
    '''
    Purpose: to plot friction ratio (Rf) and normalized friciton ratio (Fr)

    Format: 
    depth: pd.DataFrame, size = n x 1
    Rf: pd.DataFrame, size = n x 1
    Fr: pd.DataFrame, size = n x 1

    Content:
    depth: depth
    Rf: friction ratio
    Fr: normalized friction ratio     
    '''

    plt.figure(figsize = (6,8))
    plt.plot(Rf, depth, linestyle = '', marker = '.', markersize = 2)
    plt.plot(Fr, depth, linestyle = '', marker = '.', markersize = 2)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.ylabel(depth.columns.to_list()[0])
    plt.legend(["Rf (unnormalized)", "Fr (normalized)"])




def plotSBTn(Rf: pd.DataFrame, Qtn:pd.DataFrame, ax:matplotlib.axes._axes.Axes) -> matplotlib.axes._axes.Axes:
    '''
    Purpose: this function plots Rf vs Qtn for one cluster

    Format:
    Rf: pd.DataFrame, n x 1
    Qtn: pd.DataFrame, n x 1
    '''

    # convert Rf to percentage
    xCoords = np.log10(Rf)
    yCoords = np.log10(Qtn)


    ax.plot(xCoords, yCoords, linestyle = "none", marker = '.', markersize = 5)

    return ax



def plotSBTnAllinOne(Fr:pd.DataFrame, Qtn:pd.DataFrame, numberClusters:int, strataIndex:pd.DataFrame, SBTnImgFileName:str) -> matplotlib.axes._axes.Axes:

    '''
    Purpose: this function plots Rf vs Qtn, for all clusters

    Format: 
    Fr: pd.DataFrame
    Qtn: pd.DataFrame
    numberClusters: int
    stratIndex: pd.DataFrame
    SBTnImgFileName: str

    Content:
    Fr: normalized friction ratio
    Qtn: normalized tip resistance
    numberClusters: number of clusters
    strataIndex: the cluster index for each point
    SBTnImgFileName: file name of the SBTn image as background

    '''

    # Edge case
    if strataIndex.empty:
        print(f"Strata Index is Empty. No plot can be shown.")
        return 

    # Set up plot
    sbtImg = mping.imread(SBTnImgFileName)

    fig, ax = plt.subplots(figsize = (6, 6))
    # set image as background, and mandate xlim and ylim
    ax.imshow(sbtImg, extent = [np.log10(0.1), np.log10(10), np.log10(1), np.log10(1000)], aspect = "auto")
    # DO NOT convert x and y axes to log-scale. 
    # Because it distorts the background image at edge, if the background image is not perfect
    # DO NOT: plt.xscale("log")
    # DO NOT: plt.yscale("log")

    plt.xlim(np.log10(0.1), np.log10(10))
    plt.ylim(np.log10(1), np.log10(1000))
    plt.xlabel("Log$_{10}$ (Normalized Friction ratio)")
    plt.ylabel("Log$_{10}$ (Q$_{tn}$)")
    plt.grid(True, which = 'both', linestyle = '--', color = 'gray')

    # determine classes
    classes = np.arange(numberClusters)

    # iterate over each class
    for currentClass in classes:
        currentRf = Fr[strataIndex.iloc[:,0] == currentClass]
        currentQtn = Qtn[strataIndex.iloc[:,0] == currentClass]

        # plot current Rf and currentQtn
        plotSBTn(currentRf, currentQtn, ax)
    
    # legend
    legends = ["Layer " + str(num) for num in classes] 
    plt.legend(legends)

    return ax
    

def plotSBTnAllinAll(Fr:pd.DataFrame, Qtn:pd.DataFrame, numberClusters:int, strataIndex:pd.DataFrame, SBTnImgFileName:str) ->list[matplotlib.axes._axes.Axes]:

    '''
    Purpose: this function plots Rf vs Qtn, for all clusters

    Format:
    Fr: pd.DataFrame, size = n x 1
    Qtn: pd.DataFrame, size = n x 1
    numberClusters: int
    strataIndex: pd.DataFrame
    SBTnImgFileName: str

    '''

    # Edge case
    if strataIndex.empty:
        print(f"Strata Index is Empty. No plot can be shown.")
        return 

    # determine classes
    classes = np.arange(numberClusters)
    fig, axes = plt.subplots(numberClusters, 1, figsize = (6, 6 * numberClusters))


    for currentClass in classes:
        currentFr = Fr[strataIndex.iloc[:,0] == currentClass]
        currentQtn = Qtn[strataIndex.iloc[:,0] == currentClass]

        # plot current Rf and currentQtn
        plotSBTn(currentFr, currentQtn, axes[currentClass])

        # Set up plot
        sbtImg = mping.imread(SBTnImgFileName)

        # set image as background, and mandate xlim and ylim
        axes[currentClass].imshow(sbtImg, extent = [np.log10(0.1), np.log10(10), np.log10(1), np.log10(1000)], aspect = "auto")
        # DO NOT convert x and y axes to log-scale. 
        # Because it distorts the background image at edge, if the background image is not perfect
        # DO NOT: plt.xscale("log")
        # DO NOT: plt.yscale("log")

        plt.xlim(np.log10(0.1), np.log10(10))
        plt.ylim(np.log10(1), np.log10(1000))
  
        axes[currentClass].set_xlabel("Log$_{10}$ (Normalized Friction ratio)")
        axes[currentClass].set_ylabel("Log$_{10}$ (Q$_{tn}$)")
        axes[currentClass].grid(True, which = 'both', linestyle = '--', color = 'gray')
        axes[currentClass].legend(["Layer " + str(currentClass)])


    return axes

# plotIc
def plotIc(depth: pd.DataFrame, Ic:pd.DataFrame,  axes: list[matplotlib.axes._axes.Axes] = None) ->list[matplotlib.axes._axes.Axes]:
    '''
    Purpose: to plot CPT Soil Behavior Type Index Ic
    Reference: Guide to Cone Penetration Test 6th Ed. 2015

    Format:
    depth: pandas.DataFrame, nx1
    Ic: pandas.DataFrame, nx1
    axes: list[matplotlib.axes._axes.Axes]

    Content:
    depth: depth
    Ic: Soil behavior type index
    axes: existing axes to plot on
    '''

    if axes is None:
        fig, axes = plt.subplots(figsize = (6,6))
        axes = [axes]



    # plot threshold

    IcThresholds = [1.31, 2.05, 2.6, 2.95, 3.6]
    maxDepth = depth.values.max()

    for IcThreshold in IcThresholds:
        xthresholds = np.array([IcThreshold, IcThreshold])
        ythresholds = np.array([0, maxDepth])

        axes[0].plot(xthresholds, ythresholds, linestyle = '--', color = 'gray')

    # note thresholds
    '''
    ax.text(1, 1, "Ic>3.6: Organic Soils - Clay\n" \
    "Ic=[2.95,3.6]: Clays - silty clay to clay\n" \
    "Ic=[2.6,2.95]: silt\n" \
    "Ic=[2.05,2.6]: Sand mixtures - silty sand to sandy silt\n" \
    "Ic=[1.31,2.05]: Sands - clean sand to silty sand\n" \
    "Ic<1.31: Gravelly sand to dense sand")
    '''

    # plot data
    
    axes[0].plot( Ic, depth, linestyle = '', marker = '.', markersize = 2)
    axes[0].invert_yaxis()
    axes[0].set_xlim(1,4)
    axes[0].set_ylabel("Depth (ft)")
    axes[0].set_xlabel("Ic")
    axes[0].grid(True)
    axes[0].set_title("Ic>3.6: Organic Soils - Clay\n" \
    "Ic=[2.95,3.6]: Clays - silty clay to clay\n" \
    "Ic=[2.6,2.95]: silt\n" \
    "Ic=[2.05,2.6]: Sand mixtures - silty sand to sandy silt\n" \
    "Ic=[1.31,2.05]: Sands - clean sand to silty sand\n" \
    "Ic<1.31: Gravelly sand to dense sand", 
    horizontalalignment = "center")


    return axes

def plotIcHistogram(Ic:pd.DataFrame, axes:list[matplotlib.axes._axes.Axes] = [None]) -> list[matplotlib.axes._axes.Axes]:
    '''
    Purpose:
    To plot histogram of Ic

    Format:
    Ic: pandas.DataFrame, nx1

    Content:
    Ic: Soil behavior type index
    
    '''
    if axes[0] == None:
        fig, axes = plt.subplots(figsize = (10,4))
        axes = [axes]
    
    # plot histogram of Ic at all depths
    
    for axis in axes:
        axis.hist(Ic, bins = 30, rwidth=0.6)
        axis.set_xlabel("Ic")
        axis.set_ylabel("Count")
        axis.grid(True)
        axis.set_title("Histogram of Ic")

        #  plot Ic threshold
        counts, bins = np.histogram(Ic, bins = 30)
        plotIcThreadholds(np.max(counts), axes)

    return axes

def plotIcThreadholds(maxValue: float, axes: matplotlib.axes._axes.Axes = None) -> list[matplotlib.axes._axes.Axes]:
    '''
    Purpose: This function plots Ic threshold
    Reference: Guide to Cone Penetration Test 6th Ed. 2015

    Format:
    maxValue: float

    Content:
    maxValue: limit of threshold
    '''
    if axes is None:
        fig, axes = plt.subplots()
        axes = [axes]

    IcThresholds = [1.31, 2.05, 2.6, 2.95, 3.6]

    for axis in axes:
        for IcThreshold in IcThresholds:
            xthresholds = np.array([IcThreshold, IcThreshold])
            ythresholds = np.array([0, maxValue])
            axis.plot(xthresholds, ythresholds, linestyle = '--', color = 'gray')

    return axes

def plotSBTnCoords(SBTnShapeCoords: list[pd.DataFrame]):
    # Note: Differentiate Polygon in matplotlib and Polygon in Shapely

    '''
    Purpose: to plot polygons for each zone

    Format:
    SBTnShapeCoords: list[pd.DataFrame], len = 9

    Content:
    SBTnShapeCoords: coordinates of each zone
    '''
    fig, ax = plt.subplots()
    ax.set_xlim(0.1, 10)
    ax.set_ylim(1, 1000)
    ax.set_xlabel('Fr')
    ax.set_ylabel('Qtn')
    plt.xscale('log')
    plt.yscale('log')
    ax.set_aspect(0.65)
    

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'olive']

    for i in np.arange(9):
        # convert DataFrame to list of (x,y)
        points = SBTnShapeCoords[i][['Fr', 'Qtn']].values
        #print(points)
        # convert to polygon
        polygon = matplotlib.patches.Polygon(points, closed =True, edgecolor = 'black', facecolor = colors[i], alpha = 0.3)

        ax.add_patch(polygon)

    return ax