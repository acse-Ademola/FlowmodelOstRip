import os
import sys

import numpy as np
from itertools import chain
import numpy_indexed as npi
from scipy.sparse import csr_matrix
import warnings
import numba as nb
from numba import int64, float64

#dirname = os.path.dirname(os.path.realpath(__file__))
#sys.path.insert(0, dirname)
from solver import Solver


class Computations():

    def __init__(self, obj):
        self.obj = obj
    
    def __getattr__(self, name):
        return getattr(self.obj, name)

    #@nb.jit(nopython=True)
    def matrixSolver(self, Amatrix, Cmatrix) -> np.array:
        return Solver(Amatrix, Cmatrix).solve()
    
    def computegL(self, g) -> np.array:
        gL = np.zeros(self.nThroats)
        cond = (g[self.tList] > 0.0) & (
            (g[self.P1array] > 0) | (self.P1array < 1)) & (
            (g[self.P2array] > 0) | (self.P2array < 1))
        cond3 = cond & (g[self.P1array] > 0) & (g[self.P2array] > 0)
        cond2 = cond & (g[self.P1array] == 0) & (g[self.P2array] > 0) & (
            self.LP2array_mod > 0)
        cond1 = cond & (g[self.P1array] > 0) & (g[self.P2array] == 0) & (
            self.LP1array_mod > 0)

        gL[cond3] = 1/(self.LP1array_mod[cond3]/g[self.P1array[cond3]] +
                    self.LTarray_mod[cond3]/g[self.tList[cond3]] + self.LP2array_mod[
                    cond3]/g[self.P2array[cond3]])
        gL[cond2] = 1/(self.LTarray_mod[cond2]/g[self.tList[cond2]] + self.LP2array_mod[
                    cond2]/g[self.P2array[cond2]])
        gL[cond1] = 1/(self.LTarray_mod[cond1]/g[self.tList[cond1]] + self.LP1array_mod[
                    cond1]/g[self.P1array[cond1]])

        return gL
    

    def isConnected(self, indPS, indTS) -> np.array:
        connected = np.zeros(self.totElements, dtype='bool')

        doneP = np.ones(self.nPores+2, dtype='bool')
        doneP[indPS] = False
        doneP[0] = False
        doneT = np.ones(self.nThroats+1, dtype='bool')
        doneT[indTS] = False

        #tin = list(self.conTToIn[~doneT[self.conTToIn]])
        tin = self.elementLists[self.isOnInletBdr[1:-1] & self.connected[1:-1]]-self.nPores
        tin = list(tin[~doneT[tin]])
        tout = self.elementLists[self.isOnOutletBdr[1:-1] & self.connected[1:-1]]
        tout = list(tout[~doneT[tout-self.nPores]])

        while True:
            try:
                conn = np.zeros(self.totElements, dtype='bool')
                doneP[0] = False
                t = tin.pop(0)
                doneT[t] = True
                conn[t+self.nPores] = True
                while True:
                    #from IPython import embed; embed()
                    
                    p = np.array([self.P1array[t-1], self.P2array[t-1]])

                    p = p[~doneP[p]]
                    doneP[p] = True
                    conn[p] = True

                    try:
                        tt = np.zeros(self.nThroats+1, dtype='bool')
                        tt[np.array([*chain(*self.PTConData[p])])] = True
                        t = self.throatList[tt[1:] & ~doneT[1:]]
                        assert t.size > 0
                        doneT[t] = True
                        conn[t+self.nPores] = True
                    except (AssertionError, IndexError):
                        try:
                            tin = np.array(tin)
                            tin = list(tin[~doneT[tin]])
                        except IndexError:
                            tin=[]
                        break
                try:
                    assert conn[tout].any()
                    connected[conn] = True
                except AssertionError:
                    pass
            except (AssertionError, IndexError):
                break

        connected = connected & self.isinsideBox
        return connected


    def isTrapped(self, i, fluid, trapped):
        try:
            assert trapped[i]
            return True
        except AssertionError:
            try:
                assert i > self.nPores
                try:
                    assert self.P1array[i-self.nPores-1]*self.P2array[i-self.nPores-1] <= 0
                    return False
                except AssertionError:
                    pp = np.array([self.P1array[i-self.nPores-1], self.P2array[i-self.nPores-1]])
                    tt = np.array([])
                    indexP, indexT = [], [i]
                    
            except AssertionError:
                pp = np.array([])
                tt = self.PTConData[i]+self.nPores
                indexP, indexT = [i], []

            try:
                assert fluid == 0
                indS =  (~trapped) & ((self.fluid==0) | (
                    self.Garray <= self.bndG2))
            except AssertionError:
                indS =  (~trapped) & (self.fluid==1)
        
            indS[[-1, 0]] = True
            indS[i] = False
            try:
                pplist = list(pp[indS[pp]])
                indS[pp[indS[pp]]] = False
            except IndexError:
                pplist = []
            try:
                ttlist = list(tt[indS[tt]])
                indS[tt[indS[tt]]] = False
            except IndexError:
                ttlist = []

            while True:
                try:
                    t = ttlist.pop(np.argmin(self.distToBoundary[ttlist]))
                    indexT.append(t)
                    pp = np.array([self.P1array[t-self.nPores-1], self.P2array[
                        t-self.nPores-1]])
                    pplist.extend(pp[indS[pp]])
                    indS[pp] = False
                except ValueError:
                    pass
                try:
                    p = pplist.pop(np.argmin(self.distToBoundary[pplist]))
                    try:
                        assert p <= 0
                        return False
                    except AssertionError:
                        indexP.append(p)
                    tt = self.PTConData[p][indS[self.PTConData[p]+self.nPores]]+self.nPores
                    ttlist.extend(tt)
                    indS[tt] = False
                except ValueError:
                    pass
                try:
                    assert len(pplist)+len(ttlist) > 0
                except AssertionError:
                    try:
                        trapped[indexP] = True
                        self.PcI[indexP] = self.capPresMax*(fluid==1) + self.capPresMin*(fluid==0) 
                    except IndexError:
                        pass
                    try:
                        trapped[indexT] = True
                        self.PcI[indexT] = self.capPresMax*(fluid==1) + self.capPresMin*(fluid==0)
                    except IndexError:
                        pass
                    
                    return True


    def getValue(self, arrr, gL):
        c = arrr[self.poreList].sum()
        indP = self.poreList[arrr[self.poreList]]
        Cmatrix = np.zeros(c)
        row, col, data = [], [], []
        nPores = self.nPores
        
        def worker(arr: np.array) -> float:
            return sum(gL[arr[arrr[arr+nPores]] - 1])
        
        cond = [*map(worker, self.PTConData[indP])]
        #cond = worker()   
        m = np.arange(c)
        row.extend(m)
        col.extend(m)
        data.extend(cond)

        arrT = arrr[self.tList] & arrr[self.P1array] & arrr[self.P2array]
        cond = -gL[arrT]
        j = npi.indices(indP, self.P1array[arrT])
        k = npi.indices(indP, self.P2array[arrT])
        row.extend(j)
        col.extend(k)
        data.extend(cond)
        row.extend(k)
        col.extend(j)
        data.extend(cond)

        # for entries on/in the inlet boundary
        arrT = arrr[self.tList] & self.isOnInletBdr[self.tList]
        arrP = self.P1array[arrT]*(arrr[self.P1array[arrT]]) +\
            self.P2array[arrT]*(arrr[self.P2array[arrT]])
        cond = gL[arrT]
        m = npi.indices(indP, arrP)

        Cmatrix = np.array([*map(lambda i: cond[m == i].sum(), range(c))])
        Amatrix = csr_matrix((data, (row, col)), shape=(c, c),
                            dtype=float)
        
        return Amatrix, Cmatrix

    def _getValue_(self, arrr, gL):
        row, col, data = [], [], []
        indP = self.poreList[arrr[self.poreList]]
        c = indP.size
        Cmatrix = np.zeros(c)
        nPores = self.nPores
        mList = dict(zip(indP, np.arange(c)))

        # throats within the calcBox
        cond1 = arrr[self.tList] & arrr[self.P1array] & arrr[self.P2array]
        indT1 = zip(self.throatList[cond1], self.P1array[cond1], self.P2array[cond1])

        # throats on the inletBdr
        cond2 = arrr[self.tList] & self.isOnInletBdr[self.tList]
        indP2 = self.P1array*(cond2 & arrr[self.P1array]) + self.P2array*(
            cond2 & arrr[self.P2array])
        indT2 = zip(self.throatList[cond2], indP2[cond2])

        # throats on the outletBdr
        cond3 = arrr[self.tList] & self.isOnOutletBdr[self.tList]
        indP3 = self.P1array*(cond3 & arrr[self.P1array]) + self.P2array*(
            cond3 & arrr[self.P2array])
        indT3 = zip(self.throatList[cond3], indP3[cond3])

        def worker1(t, P1, P2):
            cond = gL[t-1]
            P1, P2 = mList[P1], mList[P2]
            row.extend((P1, P2, P1, P2))
            col.extend((P2, P1, P1, P2))
            data.extend((-cond, -cond, cond, cond))
            return
        
        def worker2(t:int, P:int):
            cond = gL[t-1]
            P = mList[P]
            row.append(P)
            col.append(P)
            data.append(cond)
            Cmatrix[P] += cond

        def worker3(t:int, P:int):
            cond = gL[t-1]
            P = mList[P]
            row.append(P)
            col.append(P)
            data.append(cond)

        #from IPython import embed; embed()
        for arr in indT1: worker1(*arr)
        for arr in indT2: worker2(*arr)
        for arr in indT3: worker3(*arr)

        Amatrix = csr_matrix((data, (row, col)), shape=(c, c), dtype=float)
        
        return Amatrix, Cmatrix
    
    def __getValue__(self, arrr, gL):
        row, col, data = [], [], []
        indP = self.poreList[arrr[self.poreList]]
        c = indP.size
        Cmatrix = np.zeros(c)
        mList = dict(zip(indP, np.arange(c)))

        # throats within the calcBox
        cond1 = arrr[self.tList] & arrr[self.P1array] & arrr[self.P2array]
        indT1 = zip(self.throatList[cond1], self.P1array[cond1], self.P2array[cond1])

        # throats on the inletBdr
        cond2 = arrr[self.tList] & self.isOnInletBdr[self.tList]
        indP2 = self.P1array*(cond2 & arrr[self.P1array]) + self.P2array*(
            cond2 & arrr[self.P2array])
        indT2 = zip(self.throatList[cond2], indP2[cond2])

        # throats on the outletBdr
        cond3 = arrr[self.tList] & self.isOnOutletBdr[self.tList]
        indP3 = self.P1array*(cond3 & arrr[self.P1array]) + self.P2array*(
            cond3 & arrr[self.P2array])
        indT3 = zip(self.throatList[cond3], indP3[cond3])

        def worker1(t, P1, P2):
            cond = gL[t-1]
            P1, P2 = mList[P1], mList[P2]
            row.extend((P1, P2, P1, P2))
            col.extend((P2, P1, P1, P2))
            data.extend((-cond, -cond, cond, cond))
            return

        def worker2(t:int, P:int):
            cond = gL[t-1]
            P = mList[P]
            row.append(P)
            col.append(P)
            data.append(cond)
            Cmatrix[P] += cond

        def worker3(t:int, P:int):
            cond = gL[t-1]
            P = mList[P]
            row.append(P)
            col.append(P)
            data.append(cond)
        
        for arr in indT1: worker1(*arr)
        for arr in indT2: worker2(*arr)
        for arr in indT3: worker3(*arr)
        
        Amatrix = csr_matrix((data, (row, col)), shape=(c, c), dtype=float)
        
        return Amatrix, Cmatrix
    
    

    def Saturation(self, AreaWP, AreaSP):
        satWP = AreaWP/AreaSP
        num = (satWP[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()
        return num/self.totVoidVolume
    

    def computeFlowrate(self, gL):
        #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        arrPoreList = np.zeros(self.nPores+2, dtype='bool')
        arrPoreList[self.P1array[(gL > 0.0)]] = True
        arrPoreList[self.P2array[(gL > 0.0)]] = True
        indPS = self.poreList[arrPoreList[1:-1]]
        indTS = self.throatList[(gL > 0.0)]
        self.conn = self.isConnected(indPS, indTS)

        Amatrix, Cmatrix = self.__getValue__(self.conn, gL)

        #from IPython import embed; embed()

        pres = np.zeros(self.nPores+2)
        pres[self.poreList[self.isOnInletBdr[self.poreList]]] = 1.0
        pres[1:-1][self.conn[self.poreList]] = self.matrixSolver(Amatrix, Cmatrix)

        delP = np.abs(pres[self.P1array] - pres[self.P2array])
        qp = gL*delP
        qinto = qp[self.isOnInletBdr[self.tList] & self.conn[self.tList]].sum()
        qout = qp[self.isOnOutletBdr[self.tList] & self.conn[self.tList]].sum()
        try:
            assert np.isclose(qinto, qout, atol=1e-30)
            qout = (qinto+qout)/2
        except AssertionError:
            pass

        return qout
    
    def computePerm(self):
        gwL = self.computegL(self.gWPhase)
        self.qW = self.computeFlowrate(gwL)
        self.obj.connW = self.conn.copy()
        self.obj.qW = self.qW
        self.krw = self.qW/self.qwSPhase

        try:
            assert self.fluid[self.tList[self.isOnOutletBdr[self.tList]]].sum() > 0
            gnwL = self.computegL(self.gNWPhase)
            self.qNW = self.computeFlowrate(gnwL)
            self.obj.connNW = self.conn.copy()
            self.obj.qNW = self.qNW
            self.krnw = self.qNW/self.qnwSPhase
        except AssertionError:
            self.qNW, self.krnw = 0, 0
        
        self.obj.fw = self.qW/(self.qW + self.qNW)
    

    def weibull(self) -> np.array:
        randNum = self.rand(self.nPores)
        if self.delta < 0 and self.eta < 0:              # Uniform Distribution
            return self.minthetai + (self.maxthetai-self.minthetai)*randNum
        else:                                  # Weibull Distribution
            return (self.maxthetai-self.minthetai)*pow(-self.delta*np.log(
                randNum*(1.0-np.exp(-1.0/self.delta))+np.exp(-1.0/self.delta)), 
                1.0/self.eta) + self.minthetai
        
    
    def __wettabilityDistribution__(self) -> np.array:
        # compute the distribution of contact angles in the network
        contactAng = np.zeros(self.totElements)
        conAng = self.weibull()        

        print(np.array([conAng[self.poreList-1].mean(), conAng[self.poreList-1].std(),
                        conAng[self.poreList-1].min(), conAng[self.poreList-1].max()])*180/np.pi)

        if self.distModel.lower() == 'rmax':
            sortedConAng = conAng[conAng.argsort()[::-1]]
            sortedPoreIndex = self.poreList[self.Rarray[self.poreList].argsort()[::-1]]
            print('rmax')
            from IPython import embed; embed()
        elif self.distModel.lower() == 'rmin':
            sortedConAng = conAng[conAng.argsort()[::-1]]
            sortedPoreIndex = self.poreList[self.Rarray[self.poreList].argsort()]
            print('rmin')
            from IPython import embed; embed()
        else:
            cond1 = (self.fluid[self.poreList] == 0)
            cond2 = (self.fluid[self.poreList] == 1)

            '''import pandas as pd
            dfP =  pd.read_csv('~/PoreFlow/data/cCAFile_pores.dat', names=['conAng'])
            conAng = np.random.choice(dfP['conAng'].values, self.nPores, replace=False)'''

            '''
            sortedConAng = conAng[conAng.argsort()]

            conAng1 = sortedConAng[:cond1.sum()]
            poreIndex1 = self.poreList[cond1]
            self.shuffle(poreIndex1)
            self.shuffle(conAng1)
            contactAng[poreIndex1] = conAng1.copy()

            poreIndex2 = self.poreList[cond2]
            conAng2 = sortedConAng[cond1.sum():]
            self.shuffle(poreIndex2)
            self.shuffle(conAng2)
            contactAng[poreIndex2] = conAng2.copy()

            '''
            sortedPoreIndex = self.poreList.copy()
            self.shuffle(sortedPoreIndex)
            self.shuffle(conAng)
            contactAng[sortedPoreIndex] = conAng.copy()  #'''
            

        randNum = self.rand(self.nThroats)
        conda = (self.P1array > 0)
        condb = (self.P2array > 0)
        condc = (conda & condb)
        
        contactAng[self.tList[~conda]] = contactAng[self.P2array[~conda]]
        contactAng[self.tList[~condb]] = contactAng[self.P1array[~condb]]
        contactAng[self.tList[condc & (randNum > 0.5)]] = contactAng[
            self.P1array[condc & (randNum > 0.5)]]
        contactAng[self.tList[condc & (randNum <= 0.5)]] = contactAng[
            self.P2array[condc & (randNum <= 0.5)]]
        
        #contactAng[self.poreList[self.fluid[self.poreList]==0]] = 0
        #contactAng[self.tList[self.fluid[self.tList]==0]] = 0
        if not self.is_oil_inj:
            #from IPython import embed; embed() 
            randNum = self.rand((self.PcD > self.maxPc).sum())
            contactAng[self.fluid==0] = 40/180*np.pi   #*randNum         # Uniform Distribution'''
        
        
        '''if not self.is_oil_inj:
            import seaborn as sns
            from matplotlib import pyplot as plt
            import pandas as pd

            from IPython import embed; embed() 

            dfP =  pd.read_csv('~/PoreFlow/data/cCAFile_pores.dat', names=['conAng'])
            dfT =  pd.read_csv('~/PoreFlow/data/cCAFile_throats.dat', names=['conAng'])
            contactAng[self.poreList] = dfP['conAng'].values/180*np.pi
            contactAng[self.tList] = dfT['conAng'].values/180*np.pi
            sns.histplot(contactAng)
            plt.show()
            #exit()
            #from IPython import embed; embed()'''
        
        print(np.array([contactAng.mean(), contactAng.std(), contactAng.min(), contactAng.max()])*180/np.pi)
        #from IPython import embed; embed()
        thetaRecAng, thetaAdvAng = self.setContactAngles(contactAng)

        return contactAng, thetaRecAng, thetaAdvAng


    
    def setContactAngles(self, contactAng) -> np.array:
        if self.wettClass == 1:
            thetaRecAng = contactAng.copy()
            thetaAdvAng = contactAng.copy()
        elif self.wettClass == 2:
            growthExp = (np.pi+self.sepAng)/np.pi
            thetaRecAng = np.maximum(0.0, growthExp*contactAng - self.sepAng)
            thetaAdvAng = np.minimum(np.pi, growthExp*contactAng)
        elif self.wettClass == 3:
            thetaRecAng = np.zeros(contactAng.size)
            thetaAdvAng = np.zeros(contactAng.size)

            cond1 = (contactAng >= 0.38349) & (contactAng < 1.5289)
            cond2 = (contactAng >= 1.5289) & (contactAng < 2.7646)
            cond3 = (contactAng >= 2.7646)
            thetaRecAng[cond1] = (0.5*np.exp(
                0.05*contactAng[cond1]*180.0/np.pi)-1.5)*np.pi/180.0
            thetaRecAng[cond2] = 2.0*(contactAng[cond2]-1.19680)
            thetaRecAng[cond3] = np.pi

            cond4 = (contactAng >= 0.38349) & (contactAng < 1.61268)
            cond5 = (contactAng >= 1.61268) & (contactAng < 2.75805)
            cond6 = (contactAng >= 2.75805)
            thetaAdvAng[cond4] = 2.0*(contactAng[cond4]-0.38349)
            thetaAdvAng[cond5] = (181.5 - 4051.0*np.exp(
                -0.05*contactAng[cond5]*180.0/np.pi))*np.pi/180.0
            thetaAdvAng[cond6] = np.pi
        elif self.wettClass == 4:
            thetaAdvAng = contactAng.copy()
            thetaRecAng = pow(np.pi - 1.3834263 - pow(
                np.pi - thetaAdvAng + 0.004, 0.45), 1.0/0.45) - 0.004
        else:
            plusCoef = np.pi - (0.1171859*(self.sepAng**3) - 0.6614868*(
                self.sepAng**2) + 1.632065*self.sepAng)
            exponentCoef = 1.0 - (0.01502745*(self.sepAng**3) - 0.1015349*(
                self.sepAng**2) + 0.4734059*self.sepAng)
            thetaAdvAng = contactAng.copy()
            thetaRecAng = pow(plusCoef - pow(
                np.pi - thetaAdvAng + 0.004, exponentCoef), 1.0/exponentCoef) - 0.004
            
        return thetaRecAng, thetaAdvAng

    def __computeFd__(self, arrr, arrBeta) -> np.array:
        thet = self.contactAng[arrr, np.newaxis]
        cond = (arrBeta < (np.pi/2-thet))
        arr3 = np.cos(thet)*np.cos(thet + arrBeta)/np.sin(arrBeta)
        arr4 = np.pi/2 - thet - arrBeta
        arr1 = (arr3-arr4)/pow(np.cos(thet), 2)
        C1 = np.sum(arr1*cond, axis=1)

        num = 1 + np.sqrt(1 - 4*self.Garray[arrr]*C1)
        den = 1 + 2*np.sqrt(np.pi*self.Garray[arrr])

        Fd = num/den
        return Fd
    
    
    def createFilms(self, arr, arrr, halfAng, Pc, m_exists,
                m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
                recPc, m_initedApexDist):

        arrr = arrr[:, np.newaxis]
        Pc = Pc[:, np.newaxis]
        cond = (~(m_exists & m_inited) & arrr)

        try:
            assert cond.sum() > 0
            conAng = self.thetaRecAng[arr, np.newaxis] if self.is_oil_inj else self.thetaAdvAng[
                arr, np.newaxis]
            condf = cond & (conAng < (np.pi/2 - halfAng))
            assert condf.sum() > 0
            m_exists[condf] = True
            m_initedApexDist[condf] = np.maximum((self.sigma/Pc*np.cos(
                conAng+halfAng)/np.sin(halfAng))[condf], 0.0)

            advPc[condf] = np.where(m_initedApexDist[
                condf] != 0.0, self.sigma*np.cos((np.minimum(np.pi, self.thetaAdvAng[
                arr, np.newaxis])+halfAng)[condf])/(
                m_initedApexDist*np.sin(halfAng))[condf], 0.0)

            recPc[condf] = np.where(m_initedApexDist[
                condf] != 0.0, self.sigma*np.cos((np.minimum(np.pi, self.thetaRecAng[
                arr, np.newaxis])+halfAng)[condf])/(
                m_initedApexDist*np.sin(halfAng))[condf], 0.0)
            
            m_inited[condf] = True
            condu = condf & (Pc > m_initOrMaxPcHist)
            assert condu.sum() > 0
            m_initOrMinApexDistHist[condu] = m_initedApexDist[condu]
            m_initOrMaxPcHist[condu] = (Pc*condu)[condu]
        except AssertionError:
            pass


    def cornerApex(self, arr, arrr, halfAng, Pc, conAng, m_exists,
               m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
               recPc, apexDist, initedApexDist, accurat=False,
               overidetrapping=False):
        warnings.simplefilter(action='ignore', category=RuntimeWarning)

        apexDist[~m_exists & arrr] = self.MOLECULAR_LENGTH
        delta = 0.0 if accurat else self._delta
        try:
            assert not overidetrapping
            apexDist[:, arrr] = initedApexDist[:, arrr]
        except AssertionError:
            pass

        # update the apex dist and contact angle
        conAng = np.ones(m_exists.shape)*conAng[arr]
        
        # condition 1
        #print('condition 1')
        cond1a = m_exists & (advPc-delta <= Pc) & (Pc <= recPc+delta)
        cond1 = cond1a & arrr
        try:
            assert cond1.sum() > 0
            #if not self.is_oil_inj and 10761 in arr: print('  cond1  ')
            part = np.minimum(0.999999, np.maximum(
                Pc*initedApexDist*np.sin(halfAng)/self.sigma,
                -0.999999))
            hingAng = np.minimum(np.maximum(
                (np.arccos(part)-halfAng)[cond1], -self._delta),
                np.pi+self._delta)
            conAng[cond1] = np.minimum(np.maximum(hingAng, 0.0), np.pi)
            apexDist[cond1] = initedApexDist[cond1]
        except AssertionError:
            pass

        # condition 2
        #print('condition 2')
        cond2a = m_exists & ~cond1a & (Pc < advPc)
        cond2 = cond2a & arrr
        try:
            assert cond2.sum() > 0
            #if not self.is_oil_inj and 10761 in arr: print('  cond2  ')
            conAng[cond2] = (self.thetaAdvAng[arr]*cond2)[cond2]
            apexDist[cond2] = (self.sigma/Pc*np.cos(
                conAng+halfAng)/np.sin(halfAng))[cond2]

            cond2b = (apexDist < initedApexDist) & cond2
            assert cond2b.sum() > 0
            #print('  cond2b  ')
            part = Pc*initedApexDist*np.sin(halfAng)/self.sigma
            part = np.minimum(0.999999, np.maximum(part, -0.999999))
            hingAng = np.minimum(np.maximum(
                (np.arccos(part)-halfAng)[cond2b], 0.0), np.pi)
            
            conAng[cond2b] = hingAng
            apexDist[cond2b] = initedApexDist[cond2b]
        except AssertionError:
            pass

        # condition 3
        #print('  condition 3   ')
        cond3a = m_exists & ~cond1a & ~cond2a & (Pc > m_initOrMaxPcHist)
        cond3 = cond3a & arrr
        try:
            assert cond3.sum() > 0
            #if not self.is_oil_inj and 10761 in arr: print('  cond3  ')
            conAng[cond3] = np.minimum(np.pi, (self.thetaRecAng[arr]*cond3)[cond3])
            apexDist[cond3] = (self.sigma/Pc*np.cos(
                conAng+halfAng)/np.sin(halfAng))[cond3]
        except AssertionError:
            pass

        # condition 4
        #print('  condition 4  ')
        cond4a = m_exists & ~cond1 & ~cond2a & ~cond3a & (Pc > recPc)
        cond4 = (cond4a*arrr)
        try:
            assert cond4.sum() > 0
            #print('  condition 4  ')
            #if not self.is_oil_inj and 10761 in arr: print('  cond4  ')
            conAng[cond4] = (self.thetaRecAng[arr]*cond4)[cond4]
            apexDist[cond4] = (self.sigma/Pc*np.cos(conAng+halfAng)/np.sin(halfAng))[cond4]
            cond4b = cond4 & (apexDist > initedApexDist)
            cond4c = cond4 & (~cond4b) & (apexDist < m_initOrMinApexDistHist)
            try:
                assert cond4b.sum() > 0
                #print('cond4b')
                part = (Pc*initedApexDist*np.sin(halfAng)/self.sigma)
                part = np.maximum(np.minimum(part, 0.999999), -0.999999)
                hingAng = np.minimum(np.maximum((
                    np.arccos(part)-halfAng)[cond4b], 0.0), np.pi)
                conAng[cond4b] = hingAng
                apexDist[cond4b] = initedApexDist[cond4b]
            except AssertionError:
                pass
            try:
                assert cond4c.sum() > 0
                #print('cond4c')
                part = (Pc*m_initOrMinApexDistHist*np.sin(halfAng)/self.sigma)
                part = np.maximum(np.minimum(part, 0.999999), -0.999999)
                hingAng = np.minimum(np.maximum((
                    np.arccos(part)-halfAng)[cond4c], 0.0), np.pi)
                conAng[cond4c] = hingAng
                apexDist[cond4c] = m_initOrMinApexDistHist[cond4c]
            except AssertionError:
                pass
        except AssertionError:
            pass

        # condition 5
        #print('  condition 5  ')
        cond5 = m_exists & ~cond1 & ~cond2 & ~cond3a & ~cond4a
        cond5 = (cond5*arrr)
        try:
            assert cond5.sum() > 0
            if not self.is_oil_inj: print('  cond5  ')
            apexDist[cond5] = ((self.sigma/Pc)*np.cos(
                conAng+halfAng)/np.sin(halfAng))[cond5]
        except AssertionError:
            pass
        
        return conAng.T, apexDist.T

        
        

        
    def cornerApex1(self, arr, arrr, halfAng, Pc, conAng, m_exists,
               m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
               recPc, apexDist, initedApexDist, hingAng, accurat=False,
               overidetrapping=False):
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
        from IPython import embed; embed()
        try:
            apexDist[~m_exists & arrr[:, np.newaxis]] = self.MOLECULAR_LENGTH
        except IndexError:
            apexDist[~m_exists & arrr] = self.MOLECULAR_LENGTH
        
        delta = 0.0 if accurat else self._delta
        try:
            assert not overidetrapping
            apexDist[arrr] = initedApexDist[arrr]
        except AssertionError:
            pass

        # update the apex dist and contact angle
        try:
            conAng = np.array([conAng[arr]]*hingAng.shape[1]).T
        except IndexError:
            conAng = conAng[arr]
        
        # condition 1
        #print('condition 1')
        cond1a = m_exists & (advPc-delta <= Pc) & (Pc <= recPc+delta)
        try:
            cond1 = cond1a & arrr
        except ValueError:
            cond1 = cond1a & arrr[:, np.newaxis]
        try:
            assert cond1.sum() > 0
            if not self.is_oil_inj: print('  cond1  ')
            part = np.minimum(0.999999, np.maximum(
                Pc*initedApexDist*np.sin(halfAng)/self.sigma,
                -0.999999))
            hingAng[cond1] = np.minimum(np.maximum(
                (np.arccos(part)-halfAng)[cond1], -self._delta),
                np.pi+self._delta)
            conAng[cond1] = np.minimum(np.maximum(hingAng[cond1], 0.0), np.pi)
            apexDist[cond1] = initedApexDist[cond1]
        except AssertionError:
            pass
        
        # condition 2
        #print('condition 2')
        cond2a = m_exists & ~cond1a & (Pc < advPc)
        try:
            cond2 = cond2a & arrr
        except ValueError:
            cond2 = cond2a & arrr[:, np.newaxis]
        try:
            assert cond2.sum() > 0
            if not self.is_oil_inj: print('  cond2  ')
            conAng[cond2] = ((self.thetaAdvAng[arr]*cond2.T).T)[cond2]
            try:
                apexDist[cond2] = (self.sigma/Pc[cond2])*np.cos(
                    conAng[cond2]+halfAng[cond2])/np.sin(halfAng[cond2])
            except (TypeError, IndexError):
                apexDist[cond2] = (self.sigma/Pc*np.cos(
                    conAng+halfAng)/np.sin(halfAng))[cond2]

            cond2b = (apexDist < initedApexDist) & cond2
            assert cond2b.sum() > 0
            print('  cond2b  ')
            part = np.minimum(0.999999, np.maximum(
                Pc*initedApexDist*np.sin(halfAng)/self.sigma, -0.999999))
            hingAng[cond2b] = np.minimum(np.maximum(
                (np.arccos(part)-halfAng)[cond2b], 0.0), np.pi)
            
            conAng[cond2b] = hingAng[cond2b]
            apexDist[cond2b] = initedApexDist[cond2b]
        except AssertionError:
            pass

        # condition 3
        #print('  condition 3   ')
        cond3a = m_exists & ~cond1a & ~cond2a & (Pc > m_initOrMaxPcHist)
        try:
            cond3 = cond3a & arrr
        except ValueError:
            cond3 = cond3a & arrr[:, np.newaxis]
        try:
            assert cond3.sum() > 0
            if not self.is_oil_inj: print('  cond3  ')
            try:
                conAng[cond3] = np.minimum(np.pi, self.thetaRecAng[arr][cond3])
                apexDist[cond3] = (self.sigma/Pc*np.cos(
                    conAng+halfAng)/np.sin(halfAng))[cond3]
            except IndexError:
                conAng[cond3] = (np.minimum(np.pi, self.thetaRecAng[
                    arr, np.newaxis])*cond3)[cond3]
                apexDist[cond3] = (self.sigma/Pc*np.cos(
                    conAng+halfAng)/np.sin(halfAng))[cond3]
        except AssertionError:
            pass

        # condition 4
        cond4a = m_exists & ~cond1 & ~cond2a & ~cond3a & (Pc > recPc)
        try:
            cond4 = (cond4a.T*arrr).T
        except ValueError:
            cond4 = (cond4a*arrr)
        except:
            print(' condition 4  ')
        try:
            assert cond4.sum() > 0
            if not self.is_oil_inj: print('  cond4  ')
            conAng[cond4] = self.thetaRecAng[arr[cond4]]
            print('Im in condition 4')
            from IPython import embed; embed()
        except AssertionError:
            pass

        # condition 5
        #print('  condition 5  ')
        cond5 = m_exists & ~cond1 & ~cond2 & ~cond3a & ~cond4a
        try:
            cond5 = (cond5.T*arrr).T
        except ValueError:
            cond5 = (cond5*arrr)
        try:
            assert cond5.sum() > 0
            if not self.is_oil_inj: print('  cond5  ')
            apexDist[cond5] = ((self.sigma/Pc)*np.cos(
                conAng+halfAng)/np.sin(halfAng))[cond5]
        except AssertionError:
            pass
        
        return conAng, apexDist


    def calcAreaW(self, arrr, halfAng, conAng, m_exists, apexDist):
        # -- obtain corner conductance -- #
        dimlessCornerA = np.zeros(m_exists.shape)

        cond1 = m_exists & (np.abs(conAng+halfAng-np.pi/2) < 0.01)
        try:
            dimlessCornerA[cond1] = np.sin(halfAng[cond1])*np.cos(halfAng[cond1])
        except IndexError:
            dimlessCornerA[cond1] = (np.sin(halfAng)*np.cos(halfAng)*cond1)[cond1]

        cond2 = m_exists & (np.abs(conAng+halfAng-np.pi/2) >= 0.01)
        dimlessCornerA[cond2] = pow((np.sin(halfAng)/np.cos(
            conAng + halfAng))[cond2], 2.0)*(np.cos(conAng)*np.cos(
            conAng + halfAng)/np.sin(halfAng)+conAng+halfAng-np.pi/2)[cond2]
        
        cornerGstar = (np.sin(halfAng)*np.cos(halfAng)/(
            4*pow(1+np.sin(halfAng), 2))*m_exists)
        cornerG = cornerGstar.copy()
        
        cond3 = m_exists & (np.abs(conAng+halfAng-np.pi/2) > 0.01)
        cornerG[cond3] = dimlessCornerA[cond3]/(4.0*pow((1 - np.sin(
            halfAng)/np.cos(conAng + halfAng)*(
                conAng + halfAng - np.pi/2))[cond3], 2.0))

        cFactor = np.where(cornerG != 0.0, 0.364+0.28*cornerGstar/cornerG, 0.0)
        conductance = cFactor*pow(apexDist, 4)*pow(
            dimlessCornerA, 2)*cornerG/self.muw
        area = apexDist*apexDist*dimlessCornerA
    
        cornerCond = conductance.sum(axis=1)
        cornerArea = area.sum(axis=1)

        return cornerArea[arrr], cornerCond[arrr]
        
    def finitCornerApex(self, arr, arrr, halfAng, Pc, m_exists,
                    m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist,
                    advPc, recPc, apexDist, m_initedApexDist, trapped):

        cond = (m_inited | (~trapped[arr])) & m_exists
        conAng = self.thetaRecAng.copy() if self.is_oil_inj else self.thetaAdvAng.copy()
        conAng, apexDist = self.cornerApex(
            arr, arrr, halfAng, Pc, conAng, 
            cond, m_initOrMaxPcHist, m_initOrMinApexDistHist,
            advPc, recPc, apexDist, m_initedApexDist)
        
        apexDist = apexDist.T
        recPc[cond] = self.sigma*np.cos((np.minimum(np.pi, self.thetaRecAng[
            arr])+halfAng)[cond])/((apexDist*np.sin(halfAng))[cond])
        advPc[cond] = self.sigma*np.cos((np.minimum(np.pi, self.thetaAdvAng[
            arr])+halfAng)[cond])/((apexDist*np.sin(halfAng))[cond])

        cond1 = cond & (Pc > m_initOrMaxPcHist)
        m_initOrMinApexDistHist[cond1] = apexDist[cond1]
        m_inited[cond] = False
        m_initedApexDist[cond] = apexDist[cond]
        
        try:
            m_initOrMaxPcHist[cond1] = Pc[cond1]
        except (TypeError, IndexError):
            m_initOrMaxPcHist[cond1] = Pc


    def initCornerApex(self, arr, arrr, halfAng, m_exists, m_inited,
                       recPc, advPc, m_initedApexDist, trapped):
    
        cond =  (m_exists & (~trapped[arr, np.newaxis]))
        m_inited[cond] = True
        Pc = self.sigma*np.cos(np.minimum(np.pi, self.thetaRecAng[
            arr, np.newaxis]+halfAng))/(m_initedApexDist*np.sin(halfAng))
        recPc[cond & (recPc < Pc)] = Pc[cond & (recPc < Pc)]
        advPc[cond] = self.sigma*np.cos((np.minimum(np.pi, self.thetaAdvAng[
            arr, np.newaxis])+halfAng)[cond])/((m_initedApexDist*np.sin(halfAng))[cond])
        


    def writeResult(self, result_str, Pc):
        print('Sw: %7.6g  \tqW: %8.6e  \tkrw: %8.6g  \tqNW: %8.6e  \tkrnw:\
              %8.6g  \tPc: %8.6g\t %8.0f invasions' % (
              self.satW, self.qW, self.krw, self.qNW, self.krnw,
              Pc, self.totNumFill, ))
            
        if self.writeData:
            result_str+="\n%.6g,%.6e,%.6g,%.6e,%.6g,%.6g,%.0f" % (
                self.satW, self.qW, self.krw, self.qNW, self.krnw,
                Pc, self.totNumFill, )
        
        return result_str




@nb.njit((int64[:], int64[:], int64[:,:], float64[:]))
def getValue1(indP, indT, PTConData, gL):
    row: int64[:]; col:int64[:]; data:float64[:]
    
    row = [1]
    col = [1]
    data = [0.001]
    for m, p in enumerate(indP):
        #for t in PTConData[p]:
         #   if t in indT:
        cond = 0
    
        
