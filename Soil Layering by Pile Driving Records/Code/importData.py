# this function is for import data
import numpy as np
import pandas as pd

def importCsv(fileName):

    #data = np.genfromtxt(fileName, delimiter = ',')
    data = pd.read_csv(fileName)

    nRow = data.shape[0]
    nCol = data.shape[1]
    print(f"Row number is: {nRow}; Column number is: {nCol}")
    print(f"Pile number is: {nCol - 1}")

    return data, nRow, nCol

