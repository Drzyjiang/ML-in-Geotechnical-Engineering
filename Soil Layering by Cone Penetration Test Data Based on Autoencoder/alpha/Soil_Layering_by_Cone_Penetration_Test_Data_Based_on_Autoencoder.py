# %% [markdown]
# # Soil Layering by Cone Penetration Test Data Based on Autoencoder

# %% [markdown]
# This notebook aims to find a deep learning approach to automatically determine soil stratification based on cone penetration test data. <br>
# Author: Zhiyan Jiang [(linkedIn.com/in/zhiyanjiang)](http://www.linkedIn.com/in/zhiyanjiang)

# %%
'''
%load_ext autoreload
%autoreload 2
'''

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import plotly.express as px

import scipy.signal

import math

from importlib import reload
import time
import gc

from pathlib import Path

# %%
import importData 
import plotCPT
import performRandomForestClustering
import performKmeansClustering

import testMaxLeafNodes
import plot
import constants
import autoencoders
import processCriteria

# %% [markdown]
# # Step 1: configurations

# %% [markdown]
# Below is configurations for data input

# %%
cptProfileFolderPath = '..\\CPT Profiles\\'
cptProfileFolderPathObj = Path('..\\CPT Profiles\\')

# get all files with '.csv' extension
cptFiles = list(cptProfileFolderPathObj.rglob('*.csv'))
cptFileNames = []

for file in cptFiles:
    cptFileNames.append(cptProfileFolderPath + file.name)

# offsets of each CPT profiles. Positive: to shift to deeper; Negative: to shift to upper
cptProfileOffsetRowsFileName = cptProfileFolderPath + 'cptProfileOffsetRows.txt'
cptProfileOffsetRowsNameFileObj = Path(cptProfileOffsetRowsFileName)
assert cptProfileOffsetRowsNameFileObj.is_file(), f"ERROR: Cannot find cptProfileOffsetRows.txt in folder {cptProfileFolderPath}"
cptProfileOffsetRows = np.loadtxt(cptProfileOffsetRowsFileName, delimiter = ',').astype(int).tolist()

# import hyper parameters from txt file
hyperParametersFileName = cptProfileFolderPath + 'hyperParameters.txt'
hyperParametersFileNameObj = Path(hyperParametersFileName)
assert hyperParametersFileNameObj.is_file(), f"ERROR: Cannot find hyperParameters.txt in folder {cptProfileFolderPath}"
with open(hyperParametersFileName, 'r') as file:
    hyperParamMap = {}
    for line in file:
        # strip leading/tailing white space
        content = line.strip().split('=')

        assert len(content) == 2, f"ERROR: Improper format of {hyperParametersFileName} in folder {cptProfileFolderPath}"
        
        hyperParamName = content[0].strip()
        hyperParamValue = content[1].strip()
        hyperParamMap[hyperParamName] = hyperParamValue

# Number of Global Classes
numberGlobalClasses = int(hyperParamMap['numberGlobalClasses'])

# Number of layers 
numberGlobalLayers = int(hyperParamMap['numberGlobalLayers'])

# Latent vector dimension. Use a multiple of 16
embed_dim = int(hyperParamMap['embed_dim'])

# number of epoch
n_epochs = int(hyperParamMap['n_epochs'])


# %% [markdown]
# Below is configurations for neural network

# %%
# Loss early termination criterion
lossEarlyTerminationCriterion = 1e-4

# choose device
myDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Flag of whether to plot data
plotImportFlag = True

# %% [markdown]
# Below is random state configurations

# %%
# random state
randomState = 0 # or randomState = time.time()
torch.manual_seed(randomState)
torch.cuda.manual_seed(randomState)
np.random.seed(randomState)

# set deterministic algorithm for Pytorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
print(f"The following files are to be analyzed:")
print(cptFileNames)

print(f"The offset rows for each profile are below:")
print(cptProfileOffsetRows)

print(f"Hyperparameter 'numberGlobalClasses' is: {numberGlobalClasses}")
print(f"HyperParameter 'numberGlobalLayers' is: {numberGlobalLayers}")

print(f"Embedding dimension is: {embed_dim}")
print(f"Number of epoch is: {n_epochs}")
print(f"Training device is: {myDevice}")
print(f"RandomState is: {randomState}")
print()

# %%


# %% [markdown]
# # Step 2: data import

# %%
# import CPT data
rawDataList = importData.importCPTs(cptFileNames, windowLength = 0, plotImportFlag = plotImportFlag)
print()

# %%
# Get CPT depth interval
cptDepthInterval = rawDataList[0]["Depth (ft)"].max() / (rawDataList[0].shape[0] - 1)

# Get depth of all CPT profiles as list
depth = []
for id in np.arange(len(cptFileNames)):
    depth.append(rawDataList[id].iloc[:,0].to_numpy())
    print(f"{cptFileNames[id]} depth shape is: {rawDataList[id].shape[0]}")

print()

# %% [markdown]
# Data preparation

# %%
# Shift padTopRows so that all profiles to be shifted down
padTopRows = np.array(cptProfileOffsetRows) - np.array(cptProfileOffsetRows).min()
paddedCPTList = importData.padCPTAtTop(rawDataList, padTopRows = padTopRows)

for cptFileId in np.arange(len(cptFileNames)):
    print(f"For {cptFileNames[cptFileId]}, the following portion is padded:") 
    print(paddedCPTList[cptFileId].iloc[:padTopRows[cptFileId], :])

print()

# %%
# Pad each CPT profile at bottom, so that the length is the same
rawDataPaddedList, rawDataPaddedCPTMaskList, paddedDepth = importData.padCPTAtBottom(paddedCPTList, padVals = None, padTopRows = cptProfileOffsetRows)

rawDataPaddedCPTMask = np.array(rawDataPaddedCPTMaskList)

# %%
# Aggregate CPT profiles
rawDataPadded = importData.stackCPT(rawDataPaddedList)


# %% [markdown]
# # Step 3: neural network

# %% [markdown]
# Prepare data for autoencoder

# %%
# Standardized depth, qc, fs
data_depth_qc_fs = rawDataPadded[:, :, :3]

data_depth_qc_fs_min = data_depth_qc_fs.min(axis = (0,1), keepdims = True)
data_depth_qc_fs_max = data_depth_qc_fs.max(axis = (0,1), keepdims = True)
data_depth_qc_fs = (data_depth_qc_fs - data_depth_qc_fs_min) / (data_depth_qc_fs_max - data_depth_qc_fs_min + constants.NONZERO_OFFSET)

print(f"Standardized shape should be [depth, qc, fs]: {data_depth_qc_fs.shape}")
print()

# %%
# Exclude depth feature
data_depth_qc_fs = data_depth_qc_fs[:,:, 1:]
print(f"After excluding depth, training data shape is: {data_depth_qc_fs.shape}")
print()

# %%
X = torch.tensor(data_depth_qc_fs, dtype = torch.float32).to(myDevice)
print(f"Training data shape is: {X.shape}")

X_mask = torch.tensor(rawDataPaddedCPTMask, dtype=torch.bool).to(myDevice)
X_mask_reverse = ~X_mask.to(bool)
print(f"After excluding depth, mask of training data shape is: {X_mask.shape}")

assert X_mask.dtype == torch.bool, "ERROR: mask type must be bool. It cannot be integer or float."
print()

# %% [markdown]
# ### Model configuration

# %%
# number of depth samples per CPT profile
SEQ_LEN = X.shape[1]

# features: depth, qc, fs
INPUT_DIM = X.shape[2]

EMBED_DIM = embed_dim

NUM_HEADS = 4

NUM_LAYERS = 1


soilTransformerModel = autoencoders.soilTransformer3(input_dim = INPUT_DIM, embed_dim = EMBED_DIM, num_heads = NUM_HEADS, 
                                       seq_len = SEQ_LEN, num_layers = NUM_LAYERS).to(myDevice)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(soilTransformerModel.parameters(), lr = 1e-3)


# %%
gc.collect()
torch.cuda.empty_cache()

print("Model training starts!")
for epoch in np.arange(n_epochs):
    soilTransformerModel.train()
    optimizer.zero_grad()
    
    # applying mask
    output = soilTransformerModel(X, src_key_padding_mask = X_mask)

    loss = criterion(output, X)
  
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch +1} / {n_epochs}, Loss: {loss.item():.4f}")
    
    # early termiantion
    if loss.item() < lossEarlyTerminationCriterion:
        print("Loss is less than 0.0001. Terminate early.")
        break

print()

# %%
# Compare original and reconstructed profiles
soilTransformerModel.eval()
with torch.no_grad():
    predictions = soilTransformerModel(X, X_mask) # [B, L, C]
 
cptFileIds = np.arange(len(cptFileNames))

for cptFileId in cptFileIds:
    original = X[cptFileId].cpu().detach().numpy()      # (length, 3)

    reconstructed = soilTransformerModel(X)[cptFileId].cpu().detach().numpy()  # (length, 3)

    plt.figure(figsize=(15, 5))
    labels = ['qc', 'fs']

    for i in range(X.shape[2]):
        plt.subplot(1, X.shape[2], i+1)
        plt.plot(original[:, i], label='Original')
        plt.plot(reconstructed[:, i], '--', label='Reconstructed')
 
 
        plt.legend()

    plt.suptitle(f"CPT Profile {cptFileNames[i]}: Original vs Reconstructed \nIf they do not match well, consider increasing n_epochs in {hyperParametersFileName} ")
    plt.tight_layout()
    
plt.show(block = False)

# %%
# get latent
selectedIds = np.arange(len(cptFileNames))
soilTransformerModel.eval()
latent = []
latentUnmasked = []
with torch.no_grad():

    for selectedId in selectedIds:
        latentCurrent = soilTransformerModel.getEmbeddings(X[selectedId].unsqueeze(0)).squeeze().cpu().detach().numpy()
        # store unmasked latent (including padded/useless portion)
        latentUnmasked.append(latentCurrent)


        # filtered out padded data points
        latentCurrent = latentCurrent[~X_mask[selectedId].to(bool).cpu().detach().numpy(),:]
        latent.append(latentCurrent)


latentUnmasked = np.array(latentUnmasked)

print()

# %% [markdown]
# # Step 4: processing all latents 

# %%
# combine all depth
depthAggregated = np.concat(depth)

# Because each profile has various seq_len, it is better to sort by the depth column
sortedIndices = depthAggregated.argsort()  
sortedIndicesInverted = np.argsort(sortedIndices)

# sort depthAggregated
depthAggregated = depthAggregated[sortedIndices]


# %%
# Prepare splitIndices to split aggregated result back into original profile sizes
splitIndices = [depth[0].shape[0]]

for id in np.arange(1, len(depth) - 1):
    splitIndices.append(splitIndices[-1] + depth[id].shape[0])

splitIndices = np.array(splitIndices)


# %%
# stack latent in list
latentAggregated = []
for i in np.arange(len(depth)):
    depthLatent = np.hstack((depth[i].reshape(-1,1), latent[i]))
    latentAggregated.append(depthLatent)

latentAggregated = np.vstack(latentAggregated)
#print(f"latentAggregated(excluding padded) shape is: {latentAggregated.shape}")

# sort latentAggregated the same way as depthAggregated
latentAggregated = latentAggregated[sortedIndices]


# %% [markdown]
# ### Apply KMeans on latentAggregated

# %%
numGlobalClassesTest = np.arange(1, 20)
performKmeansFlag = True
n_init = "auto"
configurationsToKmeans = [performKmeansFlag, numGlobalClassesTest, randomState, n_init]
plt.figure()
axes = performKmeansClustering.testKmeansK(configurationsToKmeans, latentAggregated)

axes[0].set_title(f"Look for elbow. The corresponding number of K is hyperparameter 'numberGlobalClass' in \n{cptProfileFolderPath + hyperParametersFileName}")
print()

# %%


# %%
configurationsToKmeans = [performKmeansFlag, numberGlobalClasses, randomState, n_init]
kmeansClusteringObj = performKmeansClustering.performKmeans(configurationsToKmeans, latentAggregated)
kmeansClusteringResult = performKmeansClustering.outputKmeansClustering(kmeansClusteringObj, depthAggregated, plotFlag = False)


# %%
# unzip kmeansClusteringResult
kmeansClusteringResultList = np.array(kmeansClusteringResult)

# using sortedIndicesInverted to restore the orignal sequence of latent
kmeansClusteringResultList = kmeansClusteringResultList[sortedIndicesInverted]
kmeansClusteringResultList = np.vsplit(kmeansClusteringResultList, splitIndices)

for cptProfileId in np.arange(len(cptFileNames)):
    kmeansClusteringResult = kmeansClusteringResultList[cptProfileId]


# %%
latentListForAnalysis = np.arange(len(cptFileNames))

# %%
# Pre-analysis on how many layers are needed
_, axes = plt.subplots()

for latentId in latentListForAnalysis:
    
    performRandomForestFlag = "classification"
    testMaxLeaftNodesRange = [2, 10]
    numberTrees = 1000
    dataAggregate = kmeansClusteringResultList[latentId]
    profileName = str(latentId)

    randomForestInput = [profileName, dataAggregate, numberTrees, testMaxLeaftNodesRange, randomState] 
    
    testMaxLeafNodes.testRandomForestMaxLeafNodes(performRandomForestFlag, randomForestInput, axes)
    notes = "Look for a threadhold beyond which R^2's increasing is small."

axes.set_title(notes)

# %%
# Apply random forest on kmeansClusteringResultList
randomForestUnifyClassList = []
randomForestCriteriaReducedList = []
for latentId in latentListForAnalysis:
    dataAggregate = pd.DataFrame(kmeansClusteringResultList[latentId])

    # perform random forest
    numberTrees = 2000
    maxLeafNodes = 4 # to be tuned
    performRandomForestFlag = "classification"
    randomForestInput = [dataAggregate, numberTrees, maxLeafNodes, randomState] 

    randomForestObj, randomForestResult = performRandomForestClustering.performRandomForest(performRandomForestFlag, randomForestInput, messageFlag = False)
 
    # Analyze randomforest results
    randomForestCriteria = performRandomForestClustering.getRandomForestCriteria(randomForestObj)
    randomForestCriteriaReduced = performRandomForestClustering.randomForestCriteriaMajority(randomForestCriteria)
 
    randomForestCriteriaReducedList.append(randomForestCriteriaReduced)

    # unify class based on randomForestCriteriaReduced
    randomForestUnifyClass = processCriteria.unifyClassByLayering(kmeansClusteringResultList[latentId], randomForestCriteriaReduced)
    randomForestUnifyClassList.append(randomForestUnifyClass)


# %% [markdown]
# # Step 5: Output

# %%
print()
print("Below are the FINAL OUTPUT!")
print()
for latentId in latentListForAnalysis:
    print(f"For {cptFileNames[latentId]}, layering is below:")
    print(randomForestCriteriaReducedList[latentId]["Random Forest Criteria 1D"].round(1))
    print(f"Each layer's Major Class Percentage and Entropy is below. Higher Major Class percentage or lower entropy indicates higher confidence.")
    print(randomForestUnifyClassList[latentId].round({"Major Class Percentage":2, "Entropy": 2}))
    print()
    print()

# %%
# plot layering on original CPT profile plot
for cptFileId in np.arange(len(cptFileNames)):
    rawData = rawDataList[cptFileId]
    axes = plotCPT.plotRawCPT(rawData)
    fig = axes[0].figure

    # iterate over randomForestCriteriaReducedList[cptFileId]
    qtMax = rawData["Cone resistance (tsf)"].max()
    qtMin = rawData["Cone resistance (tsf)"].min()
    fsMax = rawData["Sleeve friction (tsf)"].max()
    fsMin = rawData["Sleeve friction (tsf)"].min()

    
    fig.suptitle(f"{cptFileNames[cptFileId]}'s layering")

    for interface in randomForestCriteriaReducedList[cptFileId]["Random Forest Criteria 1D"]:
        axes[0].plot([qtMin, qtMax], [interface, interface], color = 'red', linestyle = '--', linewidth = 2, label = 'Layering')
        axes[1].plot([fsMin, fsMax], [interface, interface], color = 'red', linestyle = '--', linewidth = 2, label = 'Layering')

    # add notation of Global Class Number
    for layerId in np.arange(randomForestUnifyClassList[cptFileId].shape[0] - 1):
        yCoord = randomForestCriteriaReducedList[cptFileId]["Random Forest Criteria 1D"].iloc[layerId]
        
        globalClass = "Global Class " + str(randomForestUnifyClassList[cptFileId]["Class"].iloc[layerId])
 
        axes[0].text(qtMax, yCoord, globalClass, ha = "right", va = "bottom", color = 'r')
        axes[1].text(fsMax, yCoord, globalClass, ha = "right", va = "bottom", color = 'r')

    # Plot the last Global Class Number
    axes[0].text(qtMax, rawData["Depth (ft)"].max(), 
                 "Global Class " + str(randomForestUnifyClassList[cptFileId]["Class"].iloc[-1]), ha = "right", va = "bottom", color = 'r')
    axes[1].text(fsMax, rawData["Depth (ft)"].max(), 
                 "Global Class " + str(randomForestUnifyClassList[cptFileId]["Class"].iloc[-1]), ha = "right", va = "bottom", color = 'r')

    plt.legend(["Raw data", "Layering"])

# %%
print("Analysis is completed! Please see outcome above and the last figures.")
print("Close all figures to close this program.")

# %%
# keep figures open
plt.show()


