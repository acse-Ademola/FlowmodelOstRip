from time import time
import numpy as np
from utilities import Computations
from network import Network


class SinglePhase(Network):
    def __new__(cls, obj):
        obj.__class__ = SinglePhase
        return obj
    
    def __init__(self, obj):
        self.__areaSP__()
        self.gSP = self.__gSP__()
        self.gwSPhase = self.gSP*self.mu/self.muw
        self.gnwSPhase = self.gSP*self.mu/self.munw
        
    def __areaSP__(self):
        self.AreaSPhase = np.zeros(self.totElements)
        self.AreaSPhase[self.elementLists] = (
            (self.Rarray[1:-1])**2)/(4*self.Garray[1:-1])
    
    def __gSP__(self):
        gSP = np.zeros(self.totElements)
        gSP[1:-1] = 1/self.mu*(
            (np.pi*self.Rarray[1:-1]**4/8)*(self.Garray[1:-1] > self.bndG2)
            + (self.Rarray[1:-1]**4*0.5623*(
                (self.Garray[1:-1] >= self.bndG1) & (self.Garray[1:-1] <= self.bndG2)))
            + (self.Rarray[1:-1]**4/(16*self.Garray[1:-1])*0.6*(self.Garray[1:-1] < self.bndG1))
        )
        return gSP    


    def singlephase(self):
        compute = Computations(self)
        gLSP = compute.computegL(self.gSP)
        
        arrPoreList = np.zeros(self.nPores+2, dtype='bool')    
        arrPoreList[self.P1array[(gLSP > 0.0)]] = True
        arrPoreList[self.P2array[(gLSP > 0.0)]] = True
        indPS = self.poreList[arrPoreList[1:-1]]
        indTS = self.throatList[(gLSP > 0.0)]
        conn = compute.isConnected(indPS, indTS)

        AmatrixW, CmatrixW = compute.__getValue__(conn, gLSP)
        presSP = np.zeros(self.nPores+2)
        presSP[self.poreList[self.isOnInletBdr[self.poreList]]] = 1.0
        presSP[1:-1][conn[self.poreList]] = compute.matrixSolver(AmatrixW, CmatrixW)

        delSP = np.abs(presSP[self.P1array] - presSP[self.P2array])
        qp = gLSP*delSP

        qinto = qp[self.isOnInletBdr[self.tList] & conn[self.tList]].sum()
        qout = qp[self.isOnOutletBdr[self.tList] & conn[self.tList]].sum()

        try:
            assert np.isclose(qinto, qout, atol=1e-30)
            qout = (qinto+qout)/2
        except AssertionError:
            pass

        self.absPerm = self.mu*qout*(self.xend - self.xstart)/self.Area_
        self.qSP = qout
        self.qwSPhase = self.qSP*self.mu/self.muw
        self.qnwSPhase = self.qSP*self.mu/self.munw

        self.gwLSP = gLSP*self.mu/self.muw
        self.gnwLSP = np.zeros(self.nThroats)

        print("SPhase flowrate: w = {}, nw = {}".format(
            self.qwSPhase, self.qnwSPhase))
        print('Absolute permeability = ', self.absPerm)
        print("Time taken: {} s \n\n".format(round(time() - start, 3)))


print('------------------------------------------------------------------')
print('---------------------------Single Phase---------------------------')

start = time()
if __name__ == "__main__":
    SinglePhase().singlephase()
