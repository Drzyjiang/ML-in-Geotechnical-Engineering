import pandas as pd
import numpy as np
import shapely
from shapely.geometry import Point, Polygon
import shapely.geometry 
from constants import PSF2TSF, QTN_MAX, QTN_MIN, FR_MAX, FR_MIN, SIGMA_VO_PRIME_MIN
import matplotlib
from plotCPT import *

def calculateSigma_vo_prime(depth:pd.DataFrame, soilUnitWeight:float, waterUnitWeight:float, gwt:float) -> pd.DataFrame:
    '''
    Purpose: This function calculate effective vertical stress

    Format:
    depth: pandas.DataFrame, size = n x 1
    soilUnitWeight: float
    waterUnitWeight: float
    gwt: float
    sigma_vo_prime: pandas.DataFrame

    Content:
    depth: CPT depth
    soilUnitWeight: unit weight of soil
    waterUnitWeight: unit weight of water
    gwt: groundwater table
    sigma_vo_prime: effective vertical stress
    '''
    #sigma_vo = pd.DataFrame(columns = ["Sigma vo"])
    sigma_vo = calculateSigma_vo(depth, soilUnitWeight)

 
    hydroStaticPressure = calculateHydroStaticPressure(depth, gwt, waterUnitWeight)
  

    sigma_vo_prime = (sigma_vo.iloc[:,0].values - hydroStaticPressure.iloc[:,0].values)


    sigma_vo_prime = pd.DataFrame(sigma_vo_prime, columns = ["Sigma vo prime"])


    return sigma_vo_prime

def calculateSigma_vo(depth: pd.DataFrame, soilUnitWeight:float) -> pd.DataFrame:
    '''
    Purpose: This function calculate total vertical stress

    Format:
    depth: pandas.DataFrame, size = n x 1
    soilUnitWeight: float
    sigma_vo: pandas.DataFrame, size = n x 1

    Content:
    depth: CPT depth
    unitWeight: assumed uniform unit weight at all depths
    soilUnitWeight: unit weight of soil
    sigma_vo: effective vertical stress
    '''

    sigma_vo = depth * soilUnitWeight


    sigma_vo = sigma_vo * PSF2TSF

    sigma_vo.columns = ["Sigma vo"]
    return sigma_vo

def calculateRf(qt:pd.DataFrame, fs:pd.DataFrame) -> pd.DataFrame:
    '''
    Purpose: This function calculates friction ratio

    Format:
    qt: pandas.DataFrame, size = n x 1
    fs: pandas.DataFrame, size = n x 1
    Rf: pandas.DataFrame, size = n x 1

    Content:
    qt: tip resistance
    fs: sleeve friction
    Rf: un-normalized fricction ratio
    '''
    
    Rf = pd.DataFrame( fs/qt, columns = ["Friction ratio"])

    # convert Rf to unit of %
    Rf *=100

    return Rf

def calculateFr(depth: pd.DataFrame, qt:pd.DataFrame, fs: pd.DataFrame, soilUnitWeight: float) -> pd.DataFrame:
    '''
    Purpose: This function calculate NORMALIZED friction ratio

    Format:
    depth: pandas.DataFrame, size = n x 1
    qt: pandas.DataFrame, size = n x 1
    fs: pandas.DataFrame, size = n x 1
    soilUnitWeight: float
    Fr: pandas.DataFrame, size = n x 1

    Content:
    depth: CPT depth
    qt: tip resistance
    fs: sleeve friction
    soilUnitWeight: soil unit weight
    Fr: normalized friction ratio
    '''

    # get total vertical stress
    sigma_vo = calculateSigma_vo(depth, soilUnitWeight)

    # in case demoniator becomes negative
    denominator = (qt.iloc[:,0] - sigma_vo.iloc[:,0]).clip(lower= 1e-6)
    Fr = (fs.iloc[:,0] / denominator).to_frame()

    #Fr = (fs.iloc[:,0] / (qt.iloc[:,0] - sigma_vo.iloc[:,0])).to_frame()
    
    Fr.columns =  ["Normalized frictio ratio"]

    # convert Fr to unit of %
    Fr *=100

    # clip Fr at lower limit of 
    Fr = Fr.clip(lower = FR_MIN, upper = FR_MAX)

    return Fr


def calculateIc(qtn: pd.DataFrame, Fr: pd.DataFrame) -> pd.DataFrame:
    '''
    Purpose: This function malizes tip resistance to by effective vertical stress
    Reference: Guide to Cone Penetration Test 6th Ed. 2015

    Format:
    qtn: pandas.DataFrame, size = n x 1
    Fr: pandas.DataFrame, size = n x 1
    Ic: pandas.DataFrame, size = n x 1
 
    Content:
    data: CPT qt. Column 1: qt
    Fr: normalized sleeve friction, in percentage already
    Ic: SBTn index
    '''

    Ic = pd.DataFrame(((3.47 - np.log10(qtn.values.flatten())) **2 + (np.log10(Fr.values.flatten()) + 1.22)**2 )**0.5, columns = ["Ic"])

    # in case Ic becomes inf, cap it to non-inf max
    maxValue = Ic.replace([np.inf, -np.inf], np.nan).max().max()
    Ic = np.clip(Ic, a_min = 0, a_max = maxValue)

    return Ic

def iterateQtn(depth, qt, Fr, gwt, soilUnitWeight, waterUnitWeight, pa):
    '''
    Purpose: This function calculate Qtn by iteration, as required by CPT Guide 2015

    Format:
    depth: 
    qtn: pandas.DataFrame, size = n x 1
    sigma_vo_prime: pandas.DataFrame, size = n x 1
    Fr: pandas.DataFrame, size = n x 1
    pa: float
    gwt: float
    soilUnitWeight: float
    waterUnitWeight: float

    Content:
    qtn: CPT qtn
    Fr: sleeve friction, in unit of percentage
    sigma_vo_prime: effective vertical stress
    pa: atompheric pressure, in unit of tsf
    gwt: groundwater table
    soilUnitWeight: soil unit weight
    waterUnitWeight: soil unit weight
    '''
    row = depth.shape[0]
    previousN = pd.DataFrame(1.0, index = range(row), columns = ["previous n"]) # initial n

    sigma_vo_prime = calculateSigma_vo_prime(depth, soilUnitWeight, waterUnitWeight, gwt)

    count = 0

    while(1000 - count):
        count += 1

        qtn = calculateQtn(depth, qt, previousN, gwt, soilUnitWeight, waterUnitWeight, pa )
          
        currentIc = calculateIc(qtn, Fr)

        currentN = 0.381 * currentIc.values.flatten() + 0.05 * (sigma_vo_prime.values.flatten() /pa) - 0.15
        currentN = pd.DataFrame(currentN, columns = ["Current N"])
        #print(previousN.values.flatten() - currentN.values.flatten())

        if(((np.abs(previousN.values.flatten() - currentN.values.flatten())) > 0.01).any()):
            # update previousN
            previousN = currentN
        

        else:
            break
    
    # converged. Now can calculate qtn
    print(f"Qtn converged in {count} times")

    # According to CPT manual, n <=1.0
    previousN = previousN.clip(upper = 1.0)

    return calculateQtn(depth, qt, previousN, gwt, soilUnitWeight, waterUnitWeight, pa)



def calculateQtn(depth: pd.DataFrame, qt: pd.DataFrame, n: float, gwt:float, soilUnitWeight:float, waterUnitWeight:float, PA:float) -> pd.DataFrame:
    '''
    Purpose: This function normalizes tip resistance to by effective vertical stress

    Format:
    depth: pandas.DataFrame, size = n x 1
    qt: pandas.DataFrame, size = n x 1
    n: float
    gwt: float
    unitWeight: float
    PA: float
    qt_normalized: 

    Content:
    depth: CPT qt. Column 1: depth;
    qt: tip resitance
    n: exponential
    gwt: groundwater depth
    unitWeight: assumed uniform unit weight at all depths
    PA: atompheric pressure, 1 atm
    '''

    


    # calculate sigma_vo and sigma_vo_prime
    sigma_vo = calculateSigma_vo(depth, soilUnitWeight)
   
    
    sigma_vo_prime = calculateSigma_vo_prime(depth, soilUnitWeight, waterUnitWeight, gwt)


    # some CPT data starts at 0, which results in zero effective vertical stress
    # Thus, need to clip
    sigma_vo_prime = sigma_vo_prime.clip(lower = SIGMA_VO_PRIME_MIN) 



    cn = (PA / sigma_vo_prime.values.flatten() )** n.values.flatten()
   
    qt_normalized = ((qt.iloc[:,0] - sigma_vo.iloc[:,0] ) / PA * cn).to_frame()

    qt_normalized.columns = ["Qtn"]

    # cap qt_normalized at qt.max() or 1000
    maxValue = qt_normalized.replace([np.inf, -np.inf], np.nan).max().max()

    qt_normalized = qt_normalized.clip(lower = QTN_MIN, upper = QTN_MAX)

    return qt_normalized


def calculateHydroStaticPressure(depth:pd.DataFrame, gwt:float, waterUnitWeight:float) -> pd.DataFrame:
    '''
    Purpose: This function calculate hydrostatic pressure

    Format:
    depth: pandas.DataFrame, size = n x 1
    gwt: float
    waterUnitWeight: float
    hydroStaticPressure: pandas.DataFrame

    Content:
    depth: CPT depth
    gwt: groundwater depth
    waterUnitWeight: assumed uniform water unit weight at all depths
    hydroStaticPressure: hydrostatic pore pressure
    '''  

    mask = pd.DataFrame(False, index = range(depth.shape[0]), columns = ["mask"])

    mask = mask >= gwt
    #print(f"mask shape {mask.shape}")
    #print(f"depth shape {depth.shape}")
    
    hydroStaticPressureTemp = depth.iloc[:,0].values * waterUnitWeight * mask.iloc[:,0].values
    #print(f"hydroStaticPressureTemp shape {hydroStaticPressureTemp.shape}")


    hydroStaticPressureTemp = hydroStaticPressureTemp * PSF2TSF


    hydroStaticPressure = pd.DataFrame(hydroStaticPressureTemp,  index = range(depth.shape[0]), columns = ["HydroStatic"])
    #print(f"hydroStaticPressure shape {hydroStaticPressure.shape}")

    return hydroStaticPressure


def calculateSBTn1D(dataIc: pd.DataFrame) -> pd.DataFrame:
    '''
    Purpose: this function converts Ic to 1D SBTn index

    Format:
    Ic: pandas.DataFrame, size = n x 2
    SBTn: pandas.DataFrame, size = n x 2
    
    Content:
    Ic: soil behavior type index, [depth, Ic]
    SBTn: soil behavior zone type
    Zone 1: Sensitive, fine grained,                   Ic = N/A
    Zone 2: Organic soils - clay,                      Ic > 3.6
    Zone 3: Clays - silty clay to clay,                Ic = (2.95, 3.6]
    Zone 4: Silt mixtures - clayey silt to silty clay, Ic = (2.60, 2.95]
    Zone 5: Sand mixtures - silty sand to sandy silt,  Ic = (2.05, 2.60]
    Zone 6: Sands - clean sand to silty sand,          Ic = (1.31, 2.05]
    Zone 7: Gravelly sand to dense sand,               Ic <= 1.31
    Zone 8: Very stiff and clayey sand,                Ic = NA
    Zone 9: Very stiff, fine grained,                  Ic = NA
    '''

    # define 
    limits = [[3.6, np.inf], 
              [2.95, 3.6], 
              [2.60, 2.95],
              [2.05, 2.60],
              [1.31, 2.05],
              [-np.inf, 1.31]]
    
    row = dataIc.shape[0]
    SBTn = pd.DataFrame(index = np.arange(row), columns = [dataIc.columns.tolist()[0], "SBTn1D"])
    SBTn.iloc[:,0] = dataIc.iloc[:,0]
    offset = 2

    print(row)

    for i in np.arange(row):
        for j in np.arange(len(limits)):
            if dataIc.iloc[i,1] > limits[j][0] and dataIc.iloc[i,1] < limits[j][1]:
                SBTn.iloc[i,1] = j + offset

    # must convert type to int
    SBTn["SBTn1D"] = SBTn["SBTn1D"].astype(int)

    return SBTn

def calculateSBTn1DIc(SBTn: pd.DataFrame)->pd.DataFrame:
    '''
    Purpose:
    To convert SBTn1D result to a middle Ic value

    Format:
    SBTn: pd.DataFrame

    Content: 
    SBTn: soil behavior zone type
    Zone 1: Sensitive, fine grained,                   Ic = N/A
    Zone 2: Organic soils - clay,                      Ic > 3.6
    Zone 3: Clays - silty clay to clay,                Ic = (2.95, 3.6]
    Zone 4: Silt mixtures - clayey silt to silty clay, Ic = (2.60, 2.95]
    Zone 5: Sand mixtures - silty sand to sandy silt,  Ic = (2.05, 2.60]
    Zone 6: Sands - clean sand to silty sand,          Ic = (1.31, 2.05]
    Zone 7: Gravelly sand to dense sand,               Ic <= 1.31
    Zone 8: Very stiff and clayey sand,                Ic = NA
    Zone 9: Very stiff, fine grained,                  Ic = NA
    '''

    # map SBTn1D to the middle Ic value
    SBTnValues = [1, 2,3,4,5,6,7,8,9]
    IcValues = [4.0, 3.8, 3.275, 2.775, 2.325, 1.68, 1.155, 1.0, 1.0]
    SBTnToIcMap = {}
    for i in np.arange(len(SBTnValues)):
        SBTnToIcMap[SBTnValues[i]] = IcValues[i]
    
    SBTnIc = pd.DataFrame()
    SBTnIc["Depth (ft)"] = SBTn["Depth (ft)"]
    SBTnIc["SBTnIc"] = SBTn.iloc[:,1].map(SBTnToIcMap)

    return SBTnIc



def importSBTnChart(SBTnShapeFile: str) ->pd.DataFrame:
    '''
    Purpose: This function imports coordinates of boundaries of each zone of SBTn chart

    Format:
    SBTnShapeFile: string

    Content:
    SBTnShapeFile: coordinates of boundaries of each zone
    '''

    data = pd.read_csv(SBTnShapeFile)

    return data

def digitizeSBTnChart(data:pd.DataFrame) -> list:
    '''
    Purpose: This function group coordinates of each zone

    Format:
    data: pd.DataFrame, size = n x 4
    shapes: list of pd.DataFrame

    Content:
    data: coordinates
    shapes: 
    '''
    #print(data)

    SBTnShapeCoords = []

    for i in np.arange(1,10):
        SBTnShapeCoords.append(data[data["Zone No."] == i])

    return SBTnShapeCoords



def calculateSBTn2D(Fr:pd.DataFrame, Qtn:pd.DataFrame, SBTnShapeCoords: list[pd.DataFrame]):

    '''
    Purpose: this function converts Ic to 2D SBTn index

    Format:
    Ic: pandas.DataFrame, size = n x 1
    SBTn: pandas.DataFrame, size = n x 1
    SBTnShapeCoords: list[pandas.DataFrame], size = 9 
    
    Content:
    Fr: normalized friction ratio
    Qtn: normalized tip resistance
    '''
    
    # check input
    if Fr.shape != Qtn.shape:
        print("Friction ratio and Normalized tip resitance shapes do not match!")
        return
    
    # In SBTn chart, lower and upper limit of Qtn is [1, 1,000],
    # To enable judging if a point is inside a polygon, cap it at [QTN_MIN, QTN_MAX] 
    e = 0.001
    Qtn = Qtn.clip(lower = QTN_MIN + e, upper = QTN_MAX - e)

    # In SBT chart, lower and upper limit of Fr are [0.1, 10]
    Fr = Fr.clip(lower = FR_MIN + e, upper = FR_MAX -e)


    # Prepare polygon coorindates
    polygonList = []
    for SBTnShapeCoord in SBTnShapeCoords:
        polygonPoints = SBTnShapeCoord[['Fr', 'Qtn']].values

        # Polygon() is from Shapely, not from Matplotlib.patches
        polygon = shapely.geometry.Polygon(polygonPoints)

        polygonList.append(polygon)
    
    # Prepare output
    row = Fr.shape[0]
    SBTn2D = pd.DataFrame(index = np.arange(row), columns = ["SBTn 2D"])

    
    for i in np.arange(row):
        # point
        point = shapely.geometry.Point(Fr.iloc[i,0], Qtn.iloc[i,0])
        
        # iterate over polygonList
        for j in np.arange(len(polygonList)):
            if(polygonList[j].contains(point)):
                SBTn2D.iloc[i,0] = j+1

                break
    
    # ensure dtypes is integer
    SBTn2D = SBTn2D.astype(int)

    return SBTn2D

def verifySBTn2D(Fr:pd.DataFrame, Qtn:pd.DataFrame, SBTn2D:pd.DataFrame, SBTnShapeCoords: list[pd.DataFrame]):
    '''
    Purpose: to verify assignment of SBTn2D is correct by plotting each zone

    Format:
    Fr: pd.DataFrame, size = n x 1
    Qtn: pd.DataFrame, size = n x 1
    SBTn2D: pd.DataFrame, size = n x 1
    SBTnShapeCoords: list[pd.DataFrame], len = 9

    Content:
    Fr: normalized friction ratio
    Qtn: normalized tip resistance
    SBTn2D: resulting soil behavior type 
    SBTnShapeCoords: boundary coordinates of each 2D SBT zone
    '''

    # In SBTn chart, upper limit of Qtn is 1,000,
    # To enable judging if a point is inside a polygon, cap it at 999.999 
    Qtn = Qtn.clip(upper = 999.999)

    # plot base map
    ax = plotSBTnCoords(SBTnShapeCoords)

    # concatenate Fr, Qtn, SBTn2D
    combined = pd.concat([Fr, Qtn, SBTn2D], axis = 1)
    

    # plot [Fr, Qtn] for each zone
    for zoneNo in np.arange(1,10):
        currentZone = combined[SBTn2D["SBTn 2D"] == zoneNo]

        if currentZone.empty == False:
            ax.plot(currentZone.iloc[:,0], currentZone.iloc[:,1])

    print(f"To verify SBTn2D, Points in each zone shall have the same color.")