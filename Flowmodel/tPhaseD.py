import os
import sys
import warnings
from math import pi
from time import time
import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from utilities import Computations
from sPhase import SinglePhase

class TwoPhaseDrainage(SinglePhase):
    def __new__(cls, obj, writeData=False):
        obj.__class__ = TwoPhaseDrainage
        return obj
    
    def __init__(self, obj, writeData=False):
        self.do = Computations(self)
        
        self.fluid = np.zeros(self.totElements, dtype='int')
        self.fluid[-1] = 1   # already filled
        self.trappedW = np.zeros(self.totElements, dtype='bool')
        self.trappedNW = np.zeros(self.totElements, dtype='bool')
        self.contactAng, self.thetaRecAng, self.thetaAdvAng =\
            self.do.__wettabilityDistribution__()
        self.Fd_Tr = self.do.__computeFd__(self.elemTriangle, self.halfAnglesTr)
        self.Fd_Sq = self.do.__computeFd__(self.elemSquare, self.halfAnglesSq)

        self.cornExistsTr = np.zeros([self.nTriangles, 3], dtype='bool')
        self.cornExistsSq = np.zeros([self.nSquares, 4], dtype='bool')
        self.initedTr = np.zeros([self.nTriangles, 3], dtype='bool')
        self.initedSq = np.zeros([self.nSquares, 4], dtype='bool')
        self.initOrMaxPcHistTr = np.zeros([self.nTriangles, 3])
        self.initOrMaxPcHistSq = np.zeros([self.nSquares, 4])
        self.initOrMinApexDistHistTr = np.zeros([self.nTriangles, 3])
        self.initOrMinApexDistHistSq = np.zeros([self.nSquares, 4])
        self.initedApexDistTr = np.zeros([self.nTriangles, 3])
        self.initedApexDistSq = np.zeros([self.nSquares, 4])
        self.advPcTr = np.zeros([self.nTriangles, 3])
        self.advPcSq = np.zeros([self.nSquares, 4])
        self.recPcTr = np.zeros([self.nTriangles, 3])
        self.recPcSq = np.zeros([self.nSquares, 4])
        self.hingAngTr = np.zeros([self.nTriangles, 3])
        self.hingAngSq = np.zeros([self.nSquares, 4])
        
        self.__computePistonPc__()
        self.PcD = self.PistonPcRec.copy()
        self.PcI = np.zeros(self.totElements)
        self.centreEPOilInj = np.zeros(self.totElements)
        self.centreEPOilInj[self.elementLists] = 2*self.sigma*np.cos(
            self.thetaRecAng[self.elementLists])/self.Rarray[self.elementLists]
        
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        ElemToFill = self.nPores+self.conTToIn
        self.ElemToFill.update(ElemToFill)
        self.NinElemList = np.ones(self.totElements, dtype='bool')
        self.NinElemList[ElemToFill] = False

        self._cornArea = self.AreaSPhase.copy()
        self._centerArea = np.zeros(self.totElements) 
        self._cornCond = self.gwSPhase.copy()
        self._centerCond = np.zeros(self.totElements)

        self.capPresMax = 0
        self.capPresMin = 0
        self.is_oil_inj = True
        self.writeData = writeData

    @property
    def AreaWPhase(self):
        return self._cornArea
    
    @property
    def AreaNWPhase(self):
        return self._centerArea
    
    @property
    def gWPhase(self):
        return self._cornCond
    
    @property
    def gNWPhase(self):
        return self._centerCond
    
    def LookupList(self, k):
        return (self.PcD[k], k > self.nPores, -k)
    
    def drainage(self):
        start = time()
        if self.writeData: self.__writeHeadersD__()
        else: self.resultD_str = ""

        print('--------------------------------------------------------------')
        print('---------------------Two Phase Drainage Process---------------')

        self.SwTarget = max(self.finalSat, self.satW-self.dSw*0.5)
        self.PcTarget = min(self.maxPc, self.capPresMax+(
            self.minDeltaPc+abs(
             self.capPresMax)*self.deltaPcFraction)*0.1)
        self.oldPcTarget = 0

        while self.filling:
            self.oldSatW = self.satW
            self.__PDrainage__()        
            
            if (self.PcTarget > self.maxPc-0.001) or (
                 self.satW < self.finalSat+0.00001):
                self.filling = False
                break
            
            self.oldPcTarget = self.capPresMax
            self.PcTarget = min(self.maxPc+1e-7, self.PcTarget+(
                    self.minDeltaPc+abs(
                     self.PcTarget)*self.deltaPcFraction))
            self.SwTarget = max(self.finalSat-1e-15, round((
                    self.satW-self.dSw*0.75)/self.dSw)*self.dSw)

            if len(self.ElemToFill) == 0:
                self.filling = False

                while self.PcTarget < self.maxPc-0.001:
                    self.__CondTP_Drainage__()
                    self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
                    self.SwTarget = self.satW
                    gwL = self.do.computegL(self.gwP, self.gwT)
                    self.qW = self.do.computeFlowrate(gwL)
                    self.krw = self.qW/self.qwSPhase
                    if any(self.fluid[self.tList[self.isOnOutletBdr[self.tList]]] == 1):
                        gnwL = self.do.computegL(self.gNWPhase)
                        self.qNW = self.do.computeFlowrate(gnwL, phase=1)
                        self.krnw = self.qNW/self.qnwSPhase
                    else:
                        self.qNW, self.krnw = 0, 0

                    self.resultD_str = self.do.writeResult(self.resultD_str, self.capPresMax)
                    self.PcTarget = min(self.maxPc+1e-7, self.PcTarget+(
                        self.minDeltaPc+abs(
                         self.PcTarget)*self.deltaPcFraction))
                    self.capPresMax = self.PcTarget
                
                break

        if self.writeData:
            with open(self.file_name, 'a') as fQ:
                fQ.write(self.resultD_str)

        self.maxPc = self.capPresMax
        self.rpd = self.sigma/self.maxPc
        print("Number of trapped elements: ", self.trappedW.sum())
        #val, count = np.unique(self.clusterP, return_counts=True)
        #val, count = np.unique(self.clusterT, return_counts=True)
        print(self.rpd, self.sigma, self.maxPc)
        print('Time spent for the drainage process: ', time() - start)        
        print('==========================================================\n\n')
        del self.do


    def popUpdateOilInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcD[k]
        self.capPresMax = np.max([self.capPresMax, capPres])

        try:
            assert k > self.nPores
            ElemInd = k-self.nPores
            assert not self.do.isTrapped(k, 0, self.trappedW)
            self.fluid[k] = 1
            self.PistonPcRec[k] = self.centreEPOilInj[k]
            ppp = np.array([self.P1array[ElemInd-1], self.P2array[
                ElemInd-1]])
            p = ppp[(self.fluid[ppp] == 0) & ~(self.trappedW[ppp])]
            [*map(lambda i: self.do.isTrapped(i, 0, self.trappedW), p)]

            self.cntT += 1
            self.invInsideBox += self.isinsideBox[k]
            self.__update_PcD_ToFill__(p)
        except (AssertionError, IndexError):
            pass            

        try:
            assert k <= self.nPores
            assert not self.do.isTrapped(k, 0, self.trappedW)
            self.fluid[k] = 1
            self.PistonPcRec[k] = self.centreEPOilInj[k]
            thr = self.PTConData[k]+self.nPores
            thr = thr[(self.fluid[thr] == 0) & ~(self.trappedW[thr])]
            [*map(lambda i: self.do.isTrapped(i, 0, self.trappedW), thr)]

            self.cntP += 1
            self.invInsideBox += self.isinsideBox[k]
            self.__update_PcD_ToFill__(thr)
        except (AssertionError, IndexError):
            pass
    

    def __PDrainage__(self):
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        self.totNumFill = 0
        self.fillTarget = max(self.m_minNumFillings, int(
            self.m_initStepSize*(self.totElements)*(
             self.SwTarget-self.satW)))
        self.invInsideBox = 0

        while (self.PcTarget+1.0e-32 > self.capPresMax) & (
                self.satW > self.SwTarget):
            self.oldSatW = self.satW
            self.invInsideBox = 0
            self.cntT, self.cntP = 0, 0
            while (self.invInsideBox < self.fillTarget) & (
                len(self.ElemToFill) != 0) & (
                    self.PcD[self.ElemToFill[0]] <= self.PcTarget):
                self.popUpdateOilInj()
            try:
                assert (self.PcD[self.ElemToFill[0]] > self.PcTarget) & (
                        self.capPresMax < self.PcTarget)
                self.capPresMax = self.PcTarget
            except AssertionError:
                pass
            
            self.__CondTP_Drainage__()
            self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
            self.totNumFill += (self.cntP+self.cntT)
            try:
                self.fillTarget = max(self.m_minNumFillings, int(min(
                    self.fillTarget*self.m_maxFillIncrease,
                    self.m_extrapCutBack*(self.invInsideBox / (
                        self.satW-self.oldSatW))*(self.SwTarget-self.satW))))
            except OverflowError:
                pass
                
            try:
                assert self.PcD[self.ElemToFill[0]] <= self.PcTarget
            except AssertionError:
                break

        try:
            assert (self.PcD[self.ElemToFill[0]] > self.PcTarget)
            self.capPresMax = self.PcTarget
        except AssertionError:
            self.PcTarget = self.capPresMax
        
        self.__CondTP_Drainage__()
        self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
        self.do.computePerm()
        self.resultD_str = self.do.writeResult(self.resultD_str, self.capPresMax)

    
    def __computePc__(self, arrr, Fd): 
        Pc = self.sigma*(1+2*np.sqrt(pi*self.Garray[arrr]))*np.cos(
            self.contactAng[arrr])*Fd/self.Rarray[arrr]
        return Pc
    
    def __computePistonPc__(self) -> None:
        self.PistonPcRec = np.zeros(self.totElements)
        self.PistonPcRec[self.elemCircle] = 2*self.sigma*np.cos(
            self.contactAng[self.elemCircle])/self.Rarray[self.elemCircle]
        self.PistonPcRec[self.elemTriangle] = self.__computePc__(
            self.elemTriangle, self.Fd_Tr)
        self.PistonPcRec[self.elemSquare] = self.__computePc__(
            self.elemSquare, self.Fd_Sq)
        
    def __func1(self, arr):
        try:
            return self.PistonPcRec[arr[self.fluid[arr] == 1]].min()
        except ValueError:
            return 0
    
    def __func2(self, i):
        try:
            return self.PistonPcRec[i[(i > 0) & (self.fluid[i] == 1)]].min()
        except ValueError:
            return 0
    
    def __func3(self, i):
        try:
            self.ElemToFill.remove(i)
        except ValueError:
            pass

    def __update_PcD_ToFill__(self, arr) -> None:
        arrP = arr[arr <= self.nPores]
        arrT = arr[arr > self.nPores]
        try:
            thr = self.PTConData[arrP]+self.nPores
            minNeiPc = np.array([*map(lambda arr: self.__func1(arr), thr)])
            #from IPython import embed; embed()
            entryPc = np.maximum(0.999*minNeiPc+0.001*self.PistonPcRec[
                arrP], self.PistonPcRec[arrP])
            
            cond1 = self.NinElemList[arrP]
            cond2 = ~cond1 & (entryPc != self.PcD[arrP])
            try:
                assert cond1.sum() > 0
                self.PcD[arrP[cond1]] = entryPc[cond1]
                self.ElemToFill.update(arrP[cond1])
                self.NinElemList[arrP[cond1]] = False
            except AssertionError:
                pass
            try:
                assert cond2.sum() > 0
                #[self.ElemToFill.remove(p) for p in arrP[cond2]]
                [*map(lambda i: self.__func3(i), arrP[cond2])]
                self.PcD[arrP[cond2]] = entryPc[cond2]
                self.ElemToFill.update(arrP[cond2])
            except AssertionError:
                pass
        except IndexError:
            pass
        try:
            ppp = np.array([*zip(
                self.P1array[arrT-self.nPores-1], self.P2array[arrT-self.nPores-1])])
            minNeiPc = np.array([*map(lambda arr: self.__func2(arr), ppp)])
            entryPc = np.maximum(0.999*minNeiPc+0.001*self.PistonPcRec[arrT], self.PistonPcRec[arrT])
            
            cond1 = self.NinElemList[arrT]
            cond2 = ~cond1 & (entryPc != self.PcD[arrT])
            try:
                assert cond1.sum() > 0
                self.PcD[arrT[cond1]] = entryPc[cond1]
                self.ElemToFill.update(arrT[cond1])
                self.NinElemList[arrT[cond1]] = False
            except AssertionError:
                pass
            try:
                assert cond2.sum() > 0
                [*map(lambda i: self.__func3(i), arrT[cond2])]
                #[self.ElemToFill.remove(t) for t in arrT[cond2]]
                self.PcD[arrT[cond2]] = entryPc[cond2]
                self.ElemToFill.update(arrT[cond2])
            except AssertionError:
                pass          
        except IndexError:
            pass

    
    def __CondTP_Drainage__(self):
        # to suppress the FutureWarning and SettingWithCopyWarning respectively
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pd.options.mode.chained_assignment = None
        
        arrr = ((self.fluid == 1) & (~self.trappedW))
        arrrS = arrr[self.elemSquare]
        arrrT = arrr[self.elemTriangle]
        arrrC = arrr[self.elemCircle]
        
        # create films
        try:
            assert (arrrT.sum() > 0)
            Pc = self.PcD[self.elemTriangle]
            curConAng = self.contactAng.copy()
            self.do.createFilms(self.elemTriangle, arrrT, self.halfAnglesTr, Pc,
                        self.cornExistsTr, self.initedTr,
                        self.initOrMaxPcHistTr,
                        self.initOrMinApexDistHistTr, self.advPcTr,
                        self.recPcTr, self.initedApexDistTr)
            
            apexDist = np.empty_like(self.hingAngTr.T)
            conAngPT, apexDistPT = self.do.cornerApex(
                self.elemTriangle, arrrT, self.halfAnglesTr.T, self.capPresMax,
                curConAng, self.cornExistsTr.T, self.initOrMaxPcHistTr.T,
                self.initOrMinApexDistHistTr.T, self.advPcTr.T,
                self.recPcTr.T, apexDist, self.initedApexDistTr.T)
            
            self._cornArea[self.elemTriangle[arrrT]], self._cornCond[
                self.elemTriangle[arrrT]] = self.do.calcAreaW(
                arrrT, self.halfAnglesTr, conAngPT, self.cornExistsTr, apexDistPT) 
        except AssertionError:
            pass

        try:
            assert (arrrS.sum() > 0)
            Pc = self.PcD[self.elemSquare]
            curConAng = self.contactAng.copy()
            self.do.createFilms(self.elemSquare, arrrS, self.halfAnglesSq,
                           Pc, self.cornExistsSq, self.initedSq, self.initOrMaxPcHistSq,
                           self.initOrMinApexDistHistSq, self.advPcSq,
                           self.recPcSq, self.initedApexDistSq)

            apexDist = np.zeros(self.hingAngSq.T.shape)
            conAngPS, apexDistPS = self.do.cornerApex(
                self.elemSquare, arrrS, self.halfAnglesSq[:, np.newaxis], self.capPresMax,
                curConAng, self.cornExistsSq.T, self.initOrMaxPcHistSq.T,
                self.initOrMinApexDistHistSq.T, self.advPcSq.T,
                self.recPcSq.T, apexDist, self.initedApexDistSq.T)
            
            self._cornArea[self.elemSquare[arrrS]], self._cornCond[
                self.elemSquare[arrrS]] = self.do.calcAreaW(
                arrrS, self.halfAnglesSq, conAngPS, self.cornExistsSq, apexDistPS)
        except AssertionError:
            pass
        try:
            assert (arrrC.sum() > 0)
            self._cornArea[self.elemCircle[arrrC]] = 0.0
            self._cornCond[self.elemCircle[arrrC]] = 0.0
        except  AssertionError:
            pass
        
        cond = (self._cornArea > self.AreaSPhase) & arrr
        self._cornArea[cond] = self.AreaSPhase[cond]
        cond = (self._cornCond > self.gwSPhase) & arrr
        self._cornCond[cond] = self.gwSPhase[cond]

        self._centerArea[arrr] = self.AreaSPhase[arrr] - self._cornArea[arrr]
        self._centerCond[arrr] = np.where(self.AreaSPhase[arrr] != 0.0, self._centerArea[
            arrr]/self.AreaSPhase[arrr]*self.gnwSPhase[arrr], 0.0)


    def _updateCornerApex_(self):
        arrr = ((self.fluid == 1) & (~self.trappedW))
        arrrS = arrr[self.elemSquare]
        arrrT = arrr[self.elemTriangle]
       
        apexDist = np.zeros(self.cornExistsTr.T.shape)
        self.do.finitCornerApex(
            self.elemTriangle, arrrT, self.halfAnglesTr.T, self.maxPc,
            self.cornExistsTr.T, self.initedTr.T, self.initOrMaxPcHistTr.T,
            self.initOrMinApexDistHistTr.T, self.advPcTr.T,
            self.recPcTr.T, apexDist, self.initedApexDistTr.T, self.trappedW)
    
        apexDist = np.zeros(self.cornExistsSq.T.shape)
        self.do.finitCornerApex(
            self.elemSquare, arrrS, self.halfAnglesSq[:, np.newaxis], self.maxPc,
            self.cornExistsSq.T, self.initedSq.T, self.initOrMaxPcHistSq.T,
            self.initOrMinApexDistHistSq.T, self.advPcSq.T,
            self.recPcSq.T, apexDist, self.initedApexDistSq.T, self.trappedW)


    def __writeHeadersD__(self):
        self._num = 1
        result_dir = "./results_csv/"
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        while True:         
            file_name = os.path.join(result_dir, "FlowmodelOOP_"+
                                self.title+"_Drainage_"+str(self._num)+".csv")
            if os.path.isfile(file_name): self._num += 1
            else:
                break

        self.file_name = file_name
        self.resultD_str="======================================================================\n"
        self.resultD_str+="# Fluid properties:\nsigma (mN/m)  \tmu_w (cP)  \tmu_nw (cP)\n"
        self.resultD_str+="# \t%.6g\t\t%.6g\t\t%.6g" % (
            self.sigma*1000, self.muw*1000, self.munw*1000, )
        self.resultD_str+="\n# calcBox: \t %.6g \t %.6g" % (
            self.calcBox[0], self.calcBox[1], )
        self.resultD_str+="\n# Wettability:"
        self.resultD_str+="\n# model \tmintheta \tmaxtheta \tdelta \teta \tdistmodel"
        self.resultD_str+="\n# %.6g\t\t%.6g\t\t%.6g\t\t%.6g\t\t%.6g" % (
            self.wettClass, round(self.minthetai*180/np.pi,3), round(self.maxthetai*180/np.pi,3), self.delta, self.eta,) 
        self.resultD_str+=self.distModel
        self.resultD_str+="\nmintheta \tmaxtheta \tmean  \tstd"
        self.resultD_str+="\n# %3.6g\t\t%3.6g\t\t%3.6g\t\t%3.6g" % (
            round(self.contactAng.min()*180/np.pi,3), round(self.contactAng.max()*180/np.pi,3), 
            round(self.contactAng.mean()*180/np.pi,3), round(self.contactAng.std()*180/np.pi,3))
        
        self.resultD_str+="\nPorosity:  %3.6g" % (self.porosity)
        self.resultD_str+="\nMaximum pore connection:  %3.6g" % (self.maxPoreCon)
        self.resultD_str+="\nAverage pore-to-pore distance:  %3.6g" % (self.avgP2Pdist)
        self.resultD_str+="\nMean pore radius:  %3.6g" % (self.Rarray[self.poreList].mean())
        self.resultD_str+="\nAbsolute permeability:  %3.6g" % (self.absPerm)
        
        self.resultD_str+="\n======================================================================"
        self.resultD_str+="\n# Sw\t qW(m3/s)\t krw\t qNW(m3/s)\t krnw\t Pc\t Invasions"
