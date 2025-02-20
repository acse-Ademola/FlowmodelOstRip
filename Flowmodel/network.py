import os
import numpy as np
import pandas as pd
from math import sqrt, pi
from time import time
import warnings
from itertools import chain

from inputData import InputData

class Network(InputData):

    def __init__(self, inputFile):
        st = time()        
        super().__init__(inputFile)
        
        print('Reading network filesssss')

        self.MAX_ITER = 1000
        self.EPSILON = 1.0e-6
        self._delta = 1e-7
        self.MOLECULAR_LENGTH = 1.0E-10
        self.satW = 1.0
        self.mu = 0.001
        self.RAND_MAX = 2147483647
        self.bndG1 = sqrt(3)/36+0.000001
        self.bndG2 = 0.07
        self.bndG3 = sqrt(3)/36
        self.pin_ = -1
        self.pout_ = 0

        (self.sigma, self.muw, self.munw, self.ohmw, self.ohmnw,
         self.rhow, self.rhonw) = self.fluidproperties()
        
        (self.m_minNumFillings, self.m_initStepSize, self.m_extrapCutBack,
         self.m_maxFillIncrease, self.m_StableFilling) = self.satCovergence()
        
        self.title = self.network()
        self.calcBox = self.__calcBox__()
        self.SEED = self.randSeed()
        np.random.seed(self.SEED)
        self.dirname = os.path.dirname(__file__)
        self.__readNetworkFiles__()
        self.xstart = self.calcBox[0]*self.xDim
        self.xend = self.calcBox[1]*self.xDim
        self.NetworkData()
        
        self.__identifyConnectedElements__()
        self.__computeHalfAng__(self.elemTriangle)
        self.halfAnglesSq = np.array([pi/4, pi/4, pi/4, pi/4])
        self.__isinsideBox__()
        self.__isOnBdr__()
        self.__modifyLength__()
        self.__computeDistToExit__()
        self.__computeDistToBoundary__()
        self.__porosity__()

        self.elem = np.zeros(self.nPores+self.nThroats+2, dtype=object)
        self.elem[-1] = Inlet()     # create an inlet (index = -1)
        self.elem[0] = Outlet(L=self.Lnetwork)     # create an outlet (index = 0)
        self.elem[-1].neighbours = []
        self.elem[0].neighbours = []
        
        self.__elementList__()
        #[*map(lambda i: self.__elementList__(i), self.elementLists)]
        self.nTriangles = self.elemTriangle.size
        self.nSquares = self.elemSquare.size
        self.nCircles = self.elemCircle.size
        self.writeData()
        print('time taken:   ', time()-st)
        #from IPython import embed; embed()


    def __readNetworkFiles__(self):
        # from IPython import embed; embed()
        # read the network files and process data       
        Lines1 = open(self.cwd + '/' + str(self.title) + "_link1.dat").readlines()
        Lines2 = open(self.cwd + '/' + str(self.title) + "_link2.dat").readlines()
        Lines3 = open(self.cwd + '/' + str(self.title) + "_node1.dat").readlines()
        Lines4 = open(self.cwd + '/' + str(self.title) + "_node2.dat").readlines()

        Lines1 = [*map(str.split, Lines1)]
        Lines2 = [*map(str.split, Lines2)]
        Lines3 = [*map(str.split, Lines3)]
        Lines4 = [*map(str.split, Lines4)]

        self.nThroats = int(Lines1[0][0])
        arr1 = Lines3[0]
        [self.nPores, self.xDim, self.yDim, self.zDim] = [int(arr1[0]), float(arr1[1]), float(
            arr1[2]), float(arr1[3])]
        del arr1
        self.maxPoreCon = max([*map(lambda x: int(x[4]), Lines3[1:])])

        self.throat = [*map(self.__getDataT__, Lines1[1:], Lines2)]
        self.pore = [*map(self.__getDataP__, Lines3[1:], Lines4)]
        self.poreCon = [*map(self.__getPoreCon__, Lines3[1:])]
        self.poreCon.insert(0, np.array([]))
        self.throatCon = [*map(self.__getThroatCon__, Lines3[1:])]
        self.throatCon.insert(0, np.array([]))

        self.poreList = np.arange(1, int(self.nPores)+1)
        self.throatList = np.arange(1, int(self.nThroats)+1)
        self.tList = self.nPores+self.throatList
        self.Area_ = float(self.yDim) * float(self.zDim)
        self.Lnetwork = float(self.xDim)
        self.totElements = self.nPores+self.nThroats+2
        self.poreListS = np.arange(self.nPores+2)
        self.poreListS[-1] = -1
        self.elementLists = np.arange(1, self.nPores+self.nThroats+1)


    def __getDataT__(self, x, y):
        return [int(x[0]), int(x[1]), int(x[2]), float(x[3]), float(x[4]),
                float(y[3]), float(y[4]), float(y[5]), float(x[5]), float(y[6]),
                float(y[7])]


    def __getDataP__(self, x, y):
        a = 5+int(x[4])
        return [int(x[0]), float(x[1]), float(x[2]), float(x[3]), int(x[4]),
                float(y[1]), float(y[2]), float(y[3]), float(y[4]), bool(int(x[a])),
                bool(int(x[a+1]))]

    def __getPoreCon__(self, x):
        a = 5+int(x[4])
        return np.array([*map(int, x[5:a])], dtype='int')

    def __getThroatCon__(self, x):
        a = 5+int(x[4])
        return np.array([*map(int, x[a+2:])], dtype='int')

    def NetworkData(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pd.options.mode.chained_assignment = None

        PoreData = pd.DataFrame(self.pore, columns=[
            "P", "x", "y", "z", "connNum", "volume", "r", "shapeFact",
            "clayVol", "poreInletStat", "poreOutletStat"])

        ThroatData = pd.DataFrame(self.throat, columns=[
            "T", "P1", "P2", "r", "shapeFact", "LP1",
            "LP2", "LT", "lenTot", "volume", "clayVol"])

        self.PPConData = np.array(self.poreCon, dtype=object)
        self.PTConData = np.array(self.throatCon, dtype=object)
        self.P1array = ThroatData['P1'].values
        self.P2array = ThroatData['P2'].values
        self.LP1array = ThroatData['LP1'].values
        self.LP2array = ThroatData['LP2'].values
        self.LTarray = ThroatData['LT'].values
        self.x_array = np.zeros(self.nPores+2)
        self.x_array[-1] -= 1e-15
        self.x_array[0] = self.Lnetwork+1e-15
        self.x_array[1:-1] = PoreData['x'].values
        self.y_array = np.zeros(self.nPores+2)
        self.y_array[[-1, 0]] = self.yDim/2
        self.y_array[1:-1] = PoreData['y'].values
        self.z_array = np.zeros(self.nPores+2)
        self.z_array[[-1, 0]] = self.zDim/2
        self.z_array[1:-1] = PoreData['z'].values
        self.lenTotarray = ThroatData['lenTot'].values

        self.conPToIn = self.poreList[PoreData['poreInletStat'].values]
        self.conPToOut = self.poreList[PoreData['poreOutletStat'].values]
        self.conTToIn = self.throatList[(self.P1array == self.pin_) | (self.P2array == self.pin_)]
        self.conTToOut = self.throatList[(self.P1array == self.pout_) | (
            self.P2array == self.pout_)]

        self.Garray = np.zeros(self.totElements)
        self.Garray[self.poreList] = self.shapeFact(PoreData['shapeFact'].values)
        self.Garray[self.nPores+self.throatList] = self.shapeFact(ThroatData['shapeFact'].values)
        self.elemTriangle = self.elementLists[self.Garray[1:-1] <= self.bndG1]
        self.elemCircle = self.elementLists[self.Garray[1:-1] >= self.bndG2]
        self.elemSquare = self.elementLists[
            (self.Garray[1:-1] > self.bndG1)  & (self.Garray[1:-1] < self.bndG2)]

        self.volarray = np.zeros(self.totElements)
        self.volarray[self.poreList] = PoreData['volume'].values
        self.volarray[+self.nPores+self.throatList] = ThroatData['volume'].values
        
        self.Rarray = np.zeros(self.totElements)
        self.Rarray[self.poreList] = PoreData['r'].values
        self.Rarray[self.nPores+self.throatList] = ThroatData['r'].values

        self.ClayVolarray = np.zeros(self.totElements)
        self.ClayVolarray[self.poreList] = PoreData['clayVol'].values
        self.ClayVolarray[self.nPores+self.throatList] = ThroatData['clayVol'].values

        del PoreData
        del ThroatData

    def __isinsideBox__(self):
        self.isinsideBox = np.zeros(self.totElements, dtype='bool')
        self.isinsideBox[self.poreList] = (self.x_array[1:-1] >= self.xstart) & (
            self.x_array[1:-1] <= self.xend)
        self.isinsideBox[self.nPores+self.throatList] = (
            self.isinsideBox[self.P1array] | self.isinsideBox[self.P2array])
        
    def __isOnBdr__(self):
        self.isOnInletBdr = np.zeros(self.totElements, dtype='bool')
        self.isOnInletBdr[self.nPores+self.throatList] = (
            self.isinsideBox[self.nPores+self.throatList] & ((
                (~self.isinsideBox[self.P1array]) & (self.x_array[self.P1array] <= self.xstart)) | ((~self.isinsideBox[self.P2array]) & (self.x_array[self.P2array] <= self.xstart)))
        )
        self.isOnOutletBdr = np.zeros(self.totElements, dtype='bool')
        self.isOnOutletBdr[self.nPores+self.throatList] = (
            self.isinsideBox[self.nPores+self.throatList] & ((
                (~self.isinsideBox[self.P1array]) & (self.x_array[self.P1array] >= self.xend)) | ((~self.isinsideBox[self.P2array]) & (self.x_array[self.P2array] >= self.xend)))
        )

        cond1 = self.isinsideBox[self.nPores+self.throatList] & ~(self.isinsideBox[self.P2array])
        pp = self.P2array[cond1 & ((self.x_array[self.P2array] < self.xstart) | (
            self.P2array == self.pin_))]
        self.isOnInletBdr[pp] = True
        pp = self.P2array[cond1 & ((self.x_array[self.P2array] > self.xend) | (
            self.P2array == self.pout_))]
        self.isOnOutletBdr[pp] = True

        cond2 = self.isinsideBox[self.nPores+self.throatList] & (~self.isinsideBox[self.P1array])
        pp = self.P1array[cond2 & ((self.x_array[self.P1array] < self.xstart) | (
            self.P1array == self.pin_))]
        self.isOnInletBdr[pp] = True
        pp = self.P1array[cond2 & ((self.x_array[self.P1array] > self.xend) | (
            self.P1array == self.pout_))]
        self.isOnOutletBdr[pp] = True

        self.isOnBdr = self.isOnInletBdr | self.isOnOutletBdr


    def shapeFact(self, data):
        G = np.minimum(data, np.sqrt(3)/36-0.00005)*(data <= self.bndG1) + (1/16)*((
            data > self.bndG1) & (data < self.bndG2)) + (1/(4*np.pi))*(data >= self.bndG2)
        return G
    
    def rand(self, a=1):
        return np.random.randint(0, self.RAND_MAX, size=a)/self.RAND_MAX
    
    def shuffle(self, obj):
        np.random.shuffle(obj)

    def choice(self, obj, size=1):
        return np.random.choice(obj, size)
    
    def __identifyConnectedElements__(self):
        tin = list(self.throatList[(self.P1array == self.pin_) | (self.P2array == self.pin_)])
        tout = (self.P1array == self.pout_) | (self.P2array == self.pout_)
        self.connected = np.zeros(self.totElements, dtype='bool')
        self.connected[-1] = True

        doneP = np.zeros(self.nPores+2, dtype='bool')
        doneP[[-1, 0]] = True
        doneT = np.zeros(self.nThroats+1, dtype='bool')
        while True:
            indexP = np.zeros(self.nPores, dtype='bool')
            indexT = np.zeros(self.nThroats, dtype='bool')
            try:
                assert len(tin) > 0
            except AssertionError:
                break
            
            t = tin.pop(0)
            while True:
                doneT[t] = True
                indexT[t-1] = True
                p = np.array([self.P1array[t-1], self.P2array[t-1]])
                p = p[~doneP[p]]
                doneP[p] = True
                indexP[p-1] = True

                try:
                    assert p.size > 0
                    tt = np.zeros(self.throatList.size+1, dtype='bool')
                    tt[np.array([*chain(*self.PTConData[p])])] = True
                    t = self.throatList[tt[1:] & ~doneT[1:]]
                    assert t.size > 0
                except AssertionError:
                    try:
                        assert any(tout & indexT)
                        self.connected[self.poreList[indexP]] = True
                        self.connected[self.throatList[indexT]+self.nPores] = True
                        
                    except AssertionError:
                        pass

                    try:
                        assert len(tin) > 0
                        tin = np.array(tin)
                        tin = list(tin[~doneT[tin]])
                    except AssertionError:
                        break
                    break

        self.connected[-1] = False
        #from IPython import embed; embed()

    def __computeDistToExit__(self):
        self.distToExit = np.zeros(self.totElements)
        self.distToExit[-1] = self.Lnetwork
        self.distToExit[self.poreList] = self.Lnetwork - self.x_array[1:-1]
        self.distToExit[self.nPores+self.throatList] = np.minimum(
            self.distToExit[self.P1array], self.distToExit[self.P2array])
        
    def __computeDistToBoundary__(self):
        self.distToBoundary = np.zeros(self.totElements)
        self.distToBoundary[self.poreList] = np.minimum(self.x_array[
            1:-1], self.distToExit[self.poreList])
        self.distToBoundary[self.nPores+self.throatList] = np.minimum(
            self.distToBoundary[self.P1array], self.distToBoundary[self.P2array])
        
    def __elementList__(self):
        for ind in self.elementLists:
            try:
                assert ind <= self.nPores
                pp = Pore(self, ind)
                try:
                    assert pp.G <= self.bndG1
                    el = Element(Triangle(pp))
                except AssertionError:
                    try:
                        assert pp.G > self.bndG2
                        el = Element(Circle(pp))
                    except AssertionError:
                        el = Element(Square(pp))
            except AssertionError:
                tt = Throat(self, ind)
                try:
                    assert tt.G <= self.bndG1
                    el = Element(Triangle(tt))
                except AssertionError:
                    try:
                        assert tt.G > self.bndG2
                        el = Element(Circle(tt))
                    except AssertionError:
                        el = Element(Square(tt))

                try:
                    assert el.conToInlet
                    self.elem[-1].neighbours.append(el.indexOren)
                except AssertionError:
                    try:
                        assert el.conToOutlet
                        self.elem[0].neighbours.append(el.indexOren)
                    except AssertionError:
                        pass

            self.elem[el.indexOren] = el


    def __computeHalfAng__(self, arrr):
        angle = -12*np.sqrt(3)*self.Garray[arrr]
        assert (np.abs(angle) <= 1.0).all()

        beta2_min = np.arctan(2/np.sqrt(3)*np.cos((np.arccos(angle)/3)+(
            4*np.pi/3)))
        beta2_max = np.arctan(2/np.sqrt(3)*np.cos(np.arccos(angle)/3))

        randNum = self.rand(arrr.size)
        randNum = 0.5*(randNum+0.5)

        beta2 = beta2_min + (beta2_max - beta2_min)*randNum
        beta1 = 0.5*(np.arcsin((np.tan(beta2) + 4*self.Garray[arrr]) * np.sin(
            beta2) / (np.tan(beta2) - 4*self.Garray[arrr])) - beta2)
        beta3 = np.pi/2 - beta2 - beta1

        assert (beta1 <= beta2).all()
        assert (beta2 <= beta3).all()
        self.halfAnglesTr = np.column_stack((beta1, beta2, beta3))

    def __modifyLength__(self):
        cond1 = self.isinsideBox[self.nPores+self.throatList] & ~(self.isinsideBox[self.P2array])
        cond2 = self.isinsideBox[self.nPores+self.throatList] & (~self.isinsideBox[self.P1array])

        scaleFact = np.zeros(self.nThroats)
        bdr = np.zeros(self.nThroats)
        throatStart = np.zeros(self.nThroats)
        throatEnd = np.zeros(self.nThroats)

        self.LP1array_mod = self.LP1array.copy()
        self.LP2array_mod = self.LP2array.copy()
        self.LTarray_mod = self.LTarray.copy()

        try:
            assert cond1.sum() > 0
            scaleFact[cond1] = (self.x_array[self.P2array[cond1]]-self.x_array[
                self.P1array[cond1]])/(
                    self.LTarray[cond1]+self.LP1array[cond1]+self.LP2array[cond1])
            cond1a = cond1 & (self.P2array < 1)
            scaleFact[cond1a] = (self.x_array[self.P2array[cond1a]]-self.x_array[
                self.P1array[cond1a]])/abs(
                    self.x_array[self.P2array[cond1a]]-self.x_array[self.P1array[cond1a]])

            bdr[cond1 & (self.x_array[self.P2array] < self.xstart)] = self.xstart
            bdr[cond1 & (self.x_array[self.P2array] >= self.xstart)] = self.xend
            throatStart[cond1] = self.x_array[self.P1array[cond1]] + self.LP1array[
                cond1]*scaleFact[cond1]
            throatEnd[cond1] = throatStart[cond1] + self.LTarray[cond1]*scaleFact[cond1]

            cond1b = cond1 & (~cond1a) & (throatEnd > self.xstart) & (throatEnd < self.xend)
            cond1c = cond1 & (~cond1a) & (~cond1b) & (throatStart > self.xstart) & (
                throatStart < self.xend)
            cond1d = cond1 & (~cond1a) & (~cond1b) & (~cond1c)
            
            self.LP2array_mod[cond1a | cond1c | cond1d] = 0.0
            self.LP2array_mod[cond1b] = (bdr[cond1b] - throatEnd[cond1b])/scaleFact[
                cond1b]
            self.LTarray_mod[cond1c] = (bdr[cond1c] - throatStart[cond1c])/scaleFact[
                cond1c]
            self.LTarray_mod[cond1d] = 0.0
        except AssertionError:
            pass

        try:
            assert cond2.sum() > 0
            scaleFact[cond2] = (self.x_array[self.P1array[cond2]]-self.x_array[self.P2array[
                cond2]])/(self.LTarray[cond2]+self.LP1array[cond2]+self.LP2array[cond2])
            cond2a = cond2 & (self.P1array < 1)
            scaleFact[cond2a] = (self.x_array[self.P1array[cond2a]]-self.x_array[self.P2array[
                cond2a]])/abs(self.x_array[self.P1array[cond2a]]-self.x_array[self.P2array[cond2a]]
                            )

            bdr[cond2 & (self.x_array[self.P1array] < self.xstart)] = self.xstart
            bdr[cond2 & (self.x_array[self.P1array] >= self.xstart)] = self.xend
            throatStart[cond2] = self.x_array[self.P2array[cond2]] + self.LP2array[
                cond2]*scaleFact[cond2]
            throatEnd[cond2] = throatStart[cond2] + self.LTarray[cond2]*scaleFact[cond2]

            cond2b = cond2 & (~cond2a) & (throatEnd > self.xstart) & (throatEnd < self.xend)
            cond2c = cond2 & (~cond2a) & (~cond2b) & (throatStart > self.xstart) & (
                throatStart < self.xend)
            cond2d = cond2 & (~cond2a) & (~cond2b) & (~cond2c)
            
            self.LP1array_mod[cond2a | cond2c | cond2d] = 0.0
            self.LP1array_mod[cond2b] = (bdr[cond2b] - throatEnd[cond2b])/scaleFact[
                cond2b]
            self.LTarray_mod[cond2c] = (bdr[cond2c] - throatStart[cond2c])/scaleFact[
                cond2c]
            self.LTarray_mod[cond2d] = 0.0
        except AssertionError:
            pass
    
    def __porosity__(self):
        volTotal = self.volarray[self.isinsideBox].sum() 
        clayvolTotal = self.ClayVolarray[self.isinsideBox].sum()

        self.totBoxVolume = (self.xend-self.xstart)*self.Area_
        self.porosity = (volTotal+clayvolTotal)/self.totBoxVolume
        self.totVoidVolume = volTotal+clayvolTotal

    def writeData(self):
        print('porosity = ', self.porosity)
        print('maximum pore connection = ', self.maxPoreCon)
        #from IPython import embed; embed()
        delta_x = self.x_array[self.P2array]-self.x_array[self.P1array]
        delta_y = self.y_array[self.P2array]-self.y_array[self.P1array]
        delta_z = self.z_array[self.P2array]-self.z_array[self.P1array]
        self.avgP2Pdist = np.sqrt(pow(delta_x, 2) + pow(delta_y, 2) + pow(delta_z, 2)).mean()
        #print('Average pore-to-pore distance = ', np.mean(self.lenTotarray))
        print('Average pore-to-pore distance = ', self.avgP2Pdist)
        print('Mean pore radius = ', np.mean(self.Rarray[self.poreList]))


class Element(Network):
    def __new__(cls, obj):
        obj.__class__ = Element
        return obj
    
    def __init__(self, obj):
        self.fluid = 0
        self.trapped = False
        self.isinBox = self.isinsideBox[self.indexOren]
        self.onInletBdr = self.isOnInletBdr[self.indexOren]
        self.onOutletBdr = self.isOnOutletBdr[self.indexOren]
        self.isOnBdr = self.isOnInletBdr | self.isOnOutletBdr
        self.isConnected = self.connected[self.indexOren]

    @property
    def neighbours(self):  
        try:
            assert not self.isPore
            return np.array([self.P1, self.P2])
        except AssertionError:
            try:
                assert self.connT.size != 0
                return self.connT
            except AssertionError:
                return None    


class Pore(Network):
    def __new__(cls, obj, ind):
        obj.__class__ = Pore
        return obj
    
    def __init__(self, obj, ind):
        self.index = ind
        self.indexOren = self.index
        self.x = self.pore[self.index-1][1]
        self.y = self.pore[self.index-1][2]
        self.z = self.pore[self.index-1][3]
        self.connNum = self.pore[self.index-1][4]
        self.volume = self.pore[self.index-1][5]
        self.r = self.pore[self.index-1][6]
        self.G = self.pore[self.index-1][7]
        self.clayVol = self.pore[self.index-1][8]
        self.poreInletStat = self.pore[self.index-1][9]
        self.poreOutletStat = self.pore[self.index-1][10]
        self.connP = self.poreCon[self.index]
        self.connT = self.throatCon[self.index]+self.nPores
        self.isPore = True


class Throat(Network):
    def __new__(cls, obj, ind):
        obj.__class__ = Throat
        return obj
    
    def __init__(self, obj, ind):
        self.index = ind-self.nPores
        self.indexOren = ind
        self.P1 = self.throat[self.index-1][1]
        self.P2 = self.throat[self.index-1][2]
        self.r = self.throat[self.index-1][3]
        self.G = self.throat[self.index-1][4]
        self.LP1 = self.throat[self.index-1][5]
        self.LP2 = self.throat[self.index-1][6]
        self.LT = self.throat[self.index-1][7]
        self.lenTot = self.throat[self.index-1][8]
        self.volume = self.throat[self.index-1][9]
        self.clayVol = self.throat[self.index-1][10]
        
        # inlet = -1 and outlet = 0
        self.conToInlet = True if -1 in [self.P1, self.P2] else False
        self.conToOutlet =  True if 0 in [self.P1, self.P2] else False
        self.conToExit = self.conToInlet | self.conToOutlet
        self.isPore = False
        self.LP1mod = self.LP1array_mod[self.index-1]
        self.LP2mod = self.LP2array_mod[self.index-1]
        self.LTmod = self.LTarray_mod[self.index-1]
        
 
        

class Triangle:
    def __new__(cls, obj):
        obj.__class__ = Triangle
        return obj

    def __init__(self, obj):
        self.apexDist = np.zeros(3)
        self.c_exists = np.zeros(3, dtype='bool')
        self.hingAng = np.zeros(3)
        self.m_inited = np.zeros(3, dtype='bool')
        self.m_initOrMinApexDistHist = np.full(3, np.inf)
        self.m_initOrMaxPcHist = np.full(3, -np.inf)
        self.m_initedApexDist = np.zeros(3)
        
        self.indexTr = np.where(self.elemTriangle == self.indexOren)
        self.halfAng = self.halfAnglesTr[self.indexTr]


class Square:
    def __new__(cls, obj):
        obj.__class__ = Square
        return obj
    
    def __init__(self, obj):
        self.halfAng = np.array([pi/4, pi/4, pi/4, pi/4])
        self.apexDist = np.zeros(4)
        self.c_exists = np.zeros(4, dtype='bool')
        self.hingAng = np.zeros(4)
        self.m_inited = np.zeros(4, dtype='bool')
        self.m_initOrMinApexDistHist = np.full(4, np.inf)
        self.m_initOrMaxPcHist = np.full(4, -np.inf)
        self.m_initedApexDist = np.zeros(4)
        

class Circle:
    def __new__(cls, obj):
        obj.__class__ = Circle
        return obj
    
    def __init__(self, obj):
        pass
    

class Inlet:
    def __init__(self):
        self.index = -1
        self.x = 0.0
        self.indexOren = -1
        self.connected = True
        self.isinsideBox = False


class Outlet:
    def __init__(self, L):
        self.index = 0
        self.x = L
        self.indexOren = 0
        self.connected = False
        self.isinsideBox = False


#network = Network('./data/input_pnflow_bent.dat')
        

'''import cProfile
import pstats
profiler = cProfile.Profile()
profiler.enable()
self.__elementList__()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
from IPython import embed; embed()'''
