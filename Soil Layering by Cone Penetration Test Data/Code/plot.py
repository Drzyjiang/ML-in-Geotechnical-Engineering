import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib


def plotPercentile(data: pd.DataFrame, lowerPercentile: float, upperPercentile: float):
    '''
    Purpose:
    To plot lower percentile and upper percentile of data

    Format: 
    data: pd.DataFrame
    lowerPercentile: float
    upperPercentile: float

    Content:
    data: data to plot
    lowerPercentile: lower percentile
    upperPercentile: upper percentile
    '''


    dataNumpy = data.to_numpy()
    lowerPercent = np.percentile(dataNumpy, lowerPercentile, axis = 1)
    upperPercent = np.percentile(dataNumpy, upperPercentile, axis = 1)

    # not completed!


def plotMeanDeviation(data: pd.DataFrame):
    '''
    Purpose:
    This function plots the mean and standard deviation

    Format:
    data: pd.DataFrame

    Content:
    data: first column is y-axis, all the rest columns are data
    '''
    
    mean = data.mean(axis = 1, skipna = True)
    #mean = np.nanmean(dataNumpy[:,1:], axis = 1)
    plt.subplot(1, 2, 1)
    plt.plot(mean, data.iloc[:, 0])
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.ylabel("Depth [ft]")
    plt.xlabel("Arithmetic Mean")

    standardDeviation = data.std(axis = 1, skipna = True)
    plt.subplot(1, 2, 2)
    plt.plot(standardDeviation, data.iloc[:,0] )
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.ylabel("Depth [ft]")
    plt.xlabel("Standard Deviation")

    plt.subplots_adjust(wspace = 0.5)

def plotAggregate(dataAggregate: pd.DataFrame, labels: list[str], markerSize: int, axes:list[plt.Axes] = None) ->list[matplotlib.axes._axes.Axes]:
    '''
    Purpose:
    To plot aggregated data

    Format: 
    dataAggregate: pd.DataFrame
    labels: list of str
    markerSize: int
    axes: list[matplotlib.axes._axes.Axes], len must match dataAggregate.shape[1] -1

    Content:
    dataAggregate: data to plot, first column should be depth
    labels: labels
    markSize:
    axes: axes to plot on
    '''
 
    # first column must be depth
    plotsNumber = dataAggregate.shape[1] -1

 
    if axes is None:
        fig, axes = plt.subplots(1, plotsNumber)

        if plotsNumber == 1:
            axes = [axes]

        # prepare axes
        for i in np.arange(1, plotsNumber + 1):
            axes[i-1].invert_yaxis()
            axes[i-1].set_ylabel(labels[0])
            axes[i-1].set_xlabel(labels[i])
            axes[i-1].grid(True)

    elif len(axes) != plotsNumber:
        print(f"ERROR: Axes are provided, but number does not match data columns.")
        return 

    
    for i in np.arange(plotsNumber):
        axes[i].scatter(dataAggregate.iloc[:, i+1], dataAggregate.iloc[:,0], s = markerSize)



    return axes


# plot simplified strata
def plotSimplifiedStrata(simplifiedStrata, criteriaConverted):
    '''
    Purpose:
    This function plots piecewise strata index on top of raw data
    
    Format:
    simplifiedStrata: 
    '''
    # iterate over critieria
    numberLayers = criteriaConverted.shape[0]
    numberFeatures = simplifiedStrata.shape[1]


    # iterate over all features
    for j in np.arange(numberFeatures):
        plt.subplot(1, numberFeatures, j+1)

        for i in np.arange(numberLayers):
            xcoords = [simplifiedStrata[i], simplifiedStrata[i]]
            ycoords = criteriaConverted[i]

            # plot a straight line for the current layer
            plt.plot(xcoords, ycoords, label = f"Layer {i+1}", linewidth = 2)
            plt.title(f"The {j} th feature")
            plt.ylabel("Depth [ft]")
            plt.gca().invert_yaxis()
            plt.grid(True)
    
    plt.legend()
