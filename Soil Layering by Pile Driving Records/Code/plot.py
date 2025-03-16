import matplotlib.pyplot as plt
import numpy as np

def plotPercentile(data, lowerPercentile, upperPercentile):
    # This function plots the lower and upper percentile
    # data is dataframe
    dataNumpy = data.to_numpy()
    lowerPercent = np.percentile(dataNumpy, lowerPercentile, axis = 1)
    upperPercent = np.percentile(dataNumpy, upperPercentile, axis = 1)


def plotMeanDeviation(data):
    # This function plots the mean and standard deviation
    # data is a dataframe
    # first column is y-axis, all the rest columns are data

    dataNumpy = data.to_numpy()
    mean = np.nanmean(dataNumpy[:,1:], axis = 1)
    plt.subplot(1, 2, 1)
    plt.plot(mean, dataNumpy[:, 0])
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.ylabel("Depth [ft]")
    plt.xlabel("Arithmetic Mean")

    standardDeviation = np.nanstd(dataNumpy, axis = 1)
    plt.subplot(1, 2, 2)
    plt.plot(standardDeviation, dataNumpy[:,0] )
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.ylabel("Depth [ft]")
    plt.xlabel("Standard Deviation")

    plt.subplots_adjust(wspace = 0.5)

def plotAggregate(dataAggregateNan, labels, markerSize):
    # Plot aggregatedNan data
    plt.scatter(dataAggregateNan[:,1], dataAggregateNan[:,0], s = markerSize)
    plt.gca().invert_yaxis()
    plt.ylabel(labels[1])
    plt.xlabel(labels[0])
    plt.grid(True)

# plot simplified strata
def plotSimplifiedStrata(simplifiedStrata, criteriaConverted):
    # This function plots simplified strata, and divide layers based on criteriaConverted
    
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
