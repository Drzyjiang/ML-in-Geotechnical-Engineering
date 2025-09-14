import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
from plotCPT import *
from processCPT import *


def importCPTs(cptFileNames: list[str],  windowLength:float, filterTimes:int = 1, removePeaksFlag: bool = False, plotImportFlag: bool = True) ->pd.DataFrame:
    '''
    Purpose:
    Top wrapper. To import Multiple CPT files

    Format:
    cptFileNames: list[str]

    windowLength: float
    filterTimes: int
    plotImportFlag: bool
    removePeaksFlag: bool

    Content:
    cptFileNames: names of all CPT files to be imported

    windowLength: filter window length in feet
    filterTimes: how many times to apply filter  
    removePeaksFlag: whether to remove peaks
    plotImportFlag: whether to plot original and filter/peak removed data
    '''

    #rawData = pd.DataFrame(columns = ["Depth (ft)", "Cone resistance (tsf)", "Sleeve friction (tsf)", "Pore pressure u2 (psi)"], dtype = float)
    rawDataList = []

    for cptFileName in cptFileNames:
        dataTemp, nRow, nCol = importCPT(cptFileName)
        
        # plot raw data
        ax = None
        if plotImportFlag:
            ax = plotRawCPT(dataTemp, axes = None)
            plt.legend(["Raw data"])

        # apply filtering, if needed
        dataTemp = performAveraging(dataTemp, windowLength, filterTimes)

        #remove peaks, if needed
        if removePeaksFlag:
            dataTemp = removePeaks(removePeaksFlag, dataTemp)

        # plot filtered data
        if plotImportFlag and windowLength:
            plotRawCPT(dataTemp,  axes = ax)
            plt.legend(["Raw data", "Processed data"])

        # Append dataTemp to rawData
        #rawData = pd.concat([rawData, dataTemp], axis = 0)
        rawDataList.append(dataTemp)



    return rawDataList

def importCPT(fileName:str, header:int = 0):
    '''
    Purpose: to import CPT data in .csv format

    Format:
    fileName: string
    header: int
    data: pd.DataFrame
    nRow: int
    nCol: int

    Content:
    fileName: file name
    header: top rows to be neglected
    data: imported CPT data
    nRow: row number
    nCol: column number
    '''
    # CPT data format:
    # first row is header
    data = pd.read_csv(fileName, header = header)

    nRow = data.shape[0]
    nCol = data.shape[1]

    print(f"Row number is: {nRow}, and Column number is: {nCol}")

    
    return data, nRow, nCol

def padCPTAtTop(rawDataList, padTopRows:list[np.int64])->list[pd.DataFrame]:
    '''
    Purpose:
    Pad each CPT profile in a list at top by the designated values (qsPadVal, fsPadVal), so that all profiles have the same max length

    Format:
    rawDataList: list of pd.DataFrame
    padTopRows: list of np.int64
    return: list of pd.DataFrame

    Content:
    rawDataList: list of CPT profiles
    padTopRows: number of rows of shifting
             positive value: shift CPT profile towards greater depth/lower elevation
             negative value: shift CPT profile towards less depth/higher elevation
    return: list of padded CPT profiles
    '''

    # Sanity check
    if len(rawDataList) == 0:
        raise Exception("ERROR: rawDataList cannot be empty.")
    

    paddedCPTList = []
    paddedCPTMaskList = []
 
    for i in np.arange(len(rawDataList)):
        rawData = rawDataList[i]
        columns = rawData.columns
        #  pad each profile with the edge value
        # so that it will not affect standardization
        rawData = np.pad(rawData, ((padTopRows[i], 0), (0,0)), 'edge')

        rawData = pd.DataFrame(rawData, columns = columns)
        paddedCPTList.append(rawData)

    return paddedCPTList


def padCPTAtBottom(rawDataList, padVals:list[np.float64] = None, padTopRows:list[np.int64] = None)->list[pd.DataFrame]:
    '''
    Purpose:
    Pad each CPT profile in a list at bottom by edge value, so that all profiles have the same max length

    Format:
    rawDataList: list of pd.DataFrame
    padVals: list of np.float64
    return: list of pd.DataFrame

    Content:
    rawDataList: list of CPT profiles
    padVals: list of 
             depthPadVal: pad value for depth
             qPadVal: pad value for cone/tip resistance 
             fsPadVal: pad value for sleeve frition 
             porePadVal: pad value for porepressure
    return: list of padded CPT profiles
    '''
    # sanity check
    if len(rawDataList) == 0:
        raise Exception("ERROR: rawDataList cannot be empty.")
    
    if padVals != None and rawDataList[0].shape[1] != len(padVals):
        raise Exception("ERROR: length of pad values should match feature number of profiles.")

    
    if padTopRows is not None and len(rawDataList) != len(padTopRows):
        raise Exception("ERROR: padTopRows size does not match number of profiles.")

    # get max length among all profiles
    paddedDepth = None
    maxLength = 0
    for rawData in rawDataList:
        if(rawData.shape[0] > maxLength):
            paddedDepth = rawData.iloc[:,0].to_numpy()

        maxLength = max(maxLength, rawData.shape[0])

    paddedCPTList = []
    paddedCPTMaskList = []

    if padTopRows is None:
        padTopRows = [0 for i in np.arange(len(rawDataList))]

 
    for i in np.arange(len(rawDataList)):
        rawData = rawDataList[i]
        columns = rawData.columns
        rawData = rawData.to_numpy()
        currentLength = rawData.shape[0]
        currentWidth = rawData.shape[1]
        padLength = maxLength - currentLength
 
        #  pad each profile with the edge value
        # so that it will not affect standardization
        rawData = np.pad(rawData, ((0, padLength), (0,0)), 'edge')
        rawData = pd.DataFrame(rawData,  columns = columns)
        paddedCPTList.append(rawData)

        # create paddedCPTMask for use in neural network
        # only the unless postions are marked as one
        paddedCPTMask = np.ones((maxLength)).astype(bool)

        paddedCPTMask[padTopRows[i]:currentLength] = False

        paddedCPTMaskList.append(paddedCPTMask)

    return paddedCPTList, paddedCPTMaskList, paddedDepth


def trimCPTbyMinLength(rawDataList:list[pd.DataFrame]):
    '''
    Purpose:
    Trim each CPT profile in a list by the minimum length among all profiles, so that the length becomes the same

    Format:
    rawDataList: list of pd.DataFrame

    Content:
    rawDataList: list of CPT profiles
    '''

    # sanity check
    if len(rawDataList) == 0:
        raise Exception("ERROR: rawDataList cannot be empty.")

    # get minimum length 
    minLength = np.iinfo(np.int32).max

    for rawData in rawDataList:
        minLength = min(minLength, rawData.shape[0])
    
    # trim each profile
    trimmedCPTList = []
    
    for rawData in rawDataList:
        trimmedCPTList.append(rawData.iloc[:minLength, :])

    return trimmedCPTList, minLength

def trimCPTbySegmentSize(rawDataList, segmentSize: int)-> list[pd.DataFrame]:
    '''
    Purpose:
    Trim each CPT profile in a list by drop the last segment if this segment size is less than segmentSize

    Format:
    rawDataList: list of pd.DataFrame
    segmentSize: int

    Content:
    rawDataList: list of CPT profiles
    segmentSize: segment size
    '''

    # error check
    if len(rawDataList) == 0:
        raise Exception("ERROR: rawDataList cannot be empty.")

    # no trmming
    if segmentSize <=0:
        print(f"Warning: segmentsize <=0. No trimming performed.")
        return rawDataList

    trimmedCPTList = []

    for rawData in rawDataList:
        # get batchCount
        segmentCount = rawData.shape[0] // segmentSize
        totalSize = segmentCount * segmentSize

        trimmedCPTList.append(rawData.iloc[:totalSize, :])
    
    return trimmedCPTList

    
def stackCPT(rawDataList:[]) ->np.array:
    '''
    Purpose:
    convert list of 2D DataFrame to a 3D DataFrame
    
    Format:
    rawDataList: list of pd.DataFrame
    stackedCPT: np.array

    Content:
    rawDataList: list of CPT profiles
    '''

    stackedCPT = np.stack([rawData.values for rawData in rawDataList], axis = 0)

    return stackedCPT

def concatCPT(rawDataList:[]) ->np.array:
    concatCPT = np.concatenate([rawData.values for rawData in rawDataList], axis = 0)

    return concatCPT

def importGWT(fileName: str, header:int = 0):
    '''
    Purpose: this function imports groundwater table 

    Format: 
    fileName: str
    header: int
    data: pd.DataFrame

    Content:
    fileName: first row denotes CPT sounding names; 
              second row denotes GWT. The the unit is the same as in CPT data
    header: rows to be skipped
    data: GWT data
    '''

    data = pd.read_csv(fileName, header = header)

    return data


def performFFT(data:pd.DataFrame):
    '''
    Purpose:
    This function performs FFT on data

    Format: 
    data: pd.DataFrame

    Content:
    data: raw CPT data    
    '''

    frequency = np.fft.fftfreq(len(data.iloc[:,0]), d = data.iloc[1,0] - data.iloc[0,0])
    qc_fft = np.fft.fft(data.iloc[:,1])
    fs_fft = np.fft.fft(data.iloc[:,2])

    fig, axes = plt.subplots(1, 2,figsize = (14,4))

    axes[0].plot(frequency[:len(data.iloc[:,0]) // 2], np.log10(np.abs(qc_fft)[:len(data.iloc[:,0])//2]))
    
    axes[1].plot(frequency[:len(data.iloc[:,0]) // 2], np.log10(np.abs(fs_fft)[:len(data.iloc[:,0])//2]))

def performAveraging(data: pd.DataFrame, windowLength: float, filterTimes: int = 1) -> pd.DataFrame:
    '''
    Purpose:
    To perform averaging on qt and fs. First column is not averaged

    Format:
    data: pandas.DataFrame, size = n * x (x >=3)
    windowLength: int
    dataFiltered: pandas.DataFrame, size = n & x (x>=3)
    filterTimes: int

    Content:
    data: raw CPT data
    windowLength: length over which average is made
    dataFiltered: averaged data
    filterTimes: how many times need to perform averaging
    '''

    # edge case: no average
    if windowLength <=0 or filterTimes <=0: 
        return data

    # determine windowSize
    windowSize = int(windowLength / (data.iloc[1,0] - data.iloc[0,0]))

    dataFiltered = data

    for j in np.arange(filterTimes):
        for i in np.arange(data.shape[1])[1:]:
            dataFiltered.iloc[:,i] =  np.convolve(dataFiltered.iloc[:,i], np.ones(windowSize) / windowSize, mode = 'same')


    
    return dataFiltered

# Obtain qc peaks
def removePeaks(removePeaksFlag: bool, rawData: pd.DataFrame, gapLength: list[float] = [1.0, 4.0])->pd.DataFrame:
    '''
    Purpose:
    To remove peaks from raw data

    Format:
    removePeaksFlag: bool
    rawData: pd.DataFrame, size = n * x, x >=2
    gapLength: list, len = 2

    Content:
    removePeaksFlag: whether to remove peaks
    rawData: first three columns are depth, cone resistance, sleeve friction
    gapLength: min and max peak width
    '''

    if removePeaksFlag == False:
        return rawData
    
    # calculate depthInterval in feet
    depthInterval = (rawData["Depth (ft)"].max() - rawData["Depth (ft)"].min()) / (rawData.shape[0] - 1)

    prominenceMin = np.std(rawData.iloc[:,1]) #/ 2.0

    gapWindowSize = ((gapLength[0]/depthInterval).astype(int) ,(gapLength[1]/depthInterval).astype(int))

    prominenceWindowSize = (gapLength[1]/depthInterval).astype(int) 
    qcPeaks, _ = find_peaks(rawData.iloc[:,1], width = gapWindowSize, prominence = prominenceMin, wlen = prominenceWindowSize)

    if qcPeaks.size == 0:
        print("No peaks removed.")
    else:
        print("Peaks at following depths are removed:")
        print(rawData.iloc[qcPeaks])

    proms, qc_left_bases, qc_right_bases = peak_prominences(rawData.iloc[:,1], qcPeaks,  wlen = prominenceWindowSize)
    #print(rawData["Depth (ft)"].iloc[qc_left_bases])
    #print(rawData["Depth (ft)"].iloc[qc_right_bases])

    # Obtain fs peaks
    prominenceMin = np.std(rawData.iloc[:,2]) / 2.0
    fsPeaks, _ = find_peaks(rawData.iloc[:,2], width = gapWindowSize, prominence = prominenceMin, wlen = prominenceWindowSize)


    proms, fs_left_bases, fs_right_bases = peak_prominences(rawData.iloc[:,2], fsPeaks, wlen = prominenceWindowSize)


    for i in np.arange(qcPeaks.shape[0]):
        #rawData.iloc[qc_left_bases[i]:qc_right_bases[i], :] = np.nan
        rawData = rawData.drop(np.arange(qc_left_bases[i], qc_right_bases[i]))

    # reindex
    rawData = rawData.reset_index(drop = True)
    rawData = rawData.reindex(np.arange(rawData.shape[0]))

    return rawData