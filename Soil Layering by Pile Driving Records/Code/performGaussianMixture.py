# This files contains functions for Gaussian Mixture

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def fitGaussianMixture(performGaussianMixtureFlag, n_components, random_state, data):
    if performGaussianMixtureFlag == False:
        return
    
    # reshape data as (-1, )
    if len(data.shape) == 1:
        data = data.reshape(-1,1)
    
    return GaussianMixture(n_components = n_components, random_state = random_state).fit(data)

def predictGaussianMixture(gmObj, data):
    # this function predicts the class id at depth

    # reshape data as (-1, )
    if len(data.shape) == 1:
        data = data.reshape(-1,1)

    return gmObj.predict(data)

def plotGaussianMixture(depth, gmResults):
    plt.scatter(gmResults, depth)
    plt.xlabel("Class")
    plt.ylabel("Depth [ft]")
    plt.gca().invert_yaxis()
    plt.grid(True)
