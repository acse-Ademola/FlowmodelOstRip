import os
import warnings
from time import time

import numpy_indexed as npi
import numpy as np
import pandas as pd
import math
from sortedcontainers import SortedList

from utilities import Computations
from tPhaseD import TwoPhaseDrainage

class TwoPhaseImbibition(TwoPhaseDrainage):
    def __new__(cls, obj, writeData=False):
        obj.__class__ = TwoPhaseImbibition
        return obj
    
    def __init__(self, obj, writeData=False):        
        self.do = Computations(self)
    
        self.porebodyPc = np.zeros(self.totElements)
        self.snapoffPc = np.zeros(self.totElements)
        self.PistonPcAdv = np.zeros(self.totElements)
        self.pistonPc_PHing = np.zeros(self.nPores+2)
        self.fluid[[-1, 0]] = 0, 1  
        self.fillmech = np.full(self.totElements, -5)

        self._updateCornerApex_()
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        self.capPresMin = self.maxPc
        self.contactAng, self.thetaRecAng, self.thetaAdvAng =\
            self.do.__wettabilityDistribution__()
        self.__initCornerApex__()
        self.__computePistonPc__()
        self.__computePc__(self.maxPc, self.elementLists, False)

        self._areaWP = self._cornArea.copy()
        self._areaNWP = self._centerArea.copy()
        self._condWP = self._cornCond.copy()
        self._condNWP = self._centerCond.copy()

        self.is_oil_inj = False
        self.writeData = writeData

    @property
    def AreaWPhase(self):
        return self._areaWP
    
    @property
    def AreaNWPhase(self):
        return self._areaNWP
    
    @property
    def gWPhase(self):
        return self._condWP
    
    @property
    def gNWPhase(self):
        return self._condNWP
    
    @property
    def cornerArea(self):
        return self._cornArea

    @property
    def centerArea(self):
        return self._centerArea

    @property
    def cornerCond(self):
        return self._cornCond

    @property
    def centerCond(self):
        return self._centerCond

    
    def imbibition(self):
        start = time()
        if self.writeData: self.__writeHeadersI__()
        else: self.resultI_str = ""

        print('--------------------------------------------------------------')
        print('-----------------------Two Phase Imbibition Process-----------')

        self.SwTarget = min(self.finalSat, self.satW+self.dSw*0.5)
        self.PcTarget = max(self.minPc, self.capPresMin-(
            self.minDeltaPc+abs(self.capPresMin)*self.deltaPcFraction)*0.1)

        self.fillTarget = max(self.m_minNumFillings, int(
            self.m_initStepSize*(self.totElements)*(
                self.satW-self.SwTarget)))
        self.done = False
        self.createFile = True

        while self.filling:
        
            self.__PImbibition__()

            if (self.PcTarget < self.minPc+0.001) or (
                 self.satW > self.finalSat-0.00001):
                self.filling = False
                break

            self.PcTarget = max(self.minPc-1e-7, self.PcTarget-(
                self.minDeltaPc+abs(
                    self.PcTarget)*self.deltaPcFraction+1e-16))
            self.SwTarget = min(self.finalSat+1e-15, round((
                self.satW+self.dSw*0.75)/self.dSw)*self.dSw)

            if len(self.ElemToFill) == 0:
                self.filling = False
                while self.PcTarget > self.minPc+0.001:
                    self.__CondTPImbibition__()
                    self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
                    self.do.computePerm()
                    self.Pc = self.PcTarget
                    self.resultI_str = self.do.writeResult(self.resultI_str, self.capPresMin)
                    self.PcTarget = max(self.minPc-1e-10, self.PcTarget-(
                        self.minDeltaPc+abs(
                         self.PcTarget)*self.deltaPcFraction))
                    self.capPresMin = self.PcTarget
                break

        if self.writeData:
            with open(self.file_name, 'a') as fQ:
                fQ.write(self.resultI_str)

        print("Number of trapped elements: ", self.trappedNW.sum())
        print('Time spent for the imbibition process: ', time() - start)
        print('===========================================================')
        del self.do
        #from IPython import embed; embed()

    def __PImbibition__(self):
        self.totNumFill = 0

        while (self.PcTarget-1.0e-32 < self.capPresMin) & (
                self.satW <= self.SwTarget):
            self.oldSatW = self.satW
            self.invInsideBox = 0
            self.cntT, self.cntP = 0, 0
            try:
                while (self.invInsideBox < self.fillTarget) & (
                    len(self.ElemToFill) != 0) & (
                        self.PcI[self.ElemToFill[0]] >= self.PcTarget):
                    
                    self.popUpdateWaterInj()
                        
                assert (self.PcI[self.ElemToFill[0]] < self.PcTarget) & (
                        self.capPresMin > self.PcTarget)
                self.capPresMin = self.PcTarget
            except IndexError:
                self.capPresMin = min(self.capPresMin, self.PcTarget)
            except AssertionError:
                pass

            self.__CondTPImbibition__()
            self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
            self.totNumFill += (self.cntP+self.cntT)

            try:
                assert self.PcI[self.ElemToFill[0]] >= self.PcTarget
            except (AssertionError, IndexError):
                break

        try:
            assert (self.PcI[self.ElemToFill[0]] < self.PcTarget) & (
                self.capPresMin > self.PcTarget)
            self.capPresMin = self.PcTarget
        except AssertionError:
            self.PcTarget = self.capPresMin
        except IndexError:
            pass

        self.__CondTPImbibition__()
        self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
        self.do.computePerm()
        self.resultI_str = self.do.writeResult(self.resultI_str, self.capPresMin)

        
    def popUpdateWaterInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcI[k]
        self.capPresMin = np.min([self.capPresMin, capPres])

        try:
            assert k > self.nPores
            ElemInd = k-self.nPores
            assert not self.do.isTrapped(k, 1, self.trappedNW)
            #print(k, capPres, self.capPresMin)
            self.fluid[k] = 0
            self.fillmech[k] = 1*(self.PistonPcAdv[k] == capPres)+3*(
                self.snapoffPc[k] == capPres)
            self.cntT += 1
            self.invInsideBox += self.isinsideBox[k]

            ppp = np.array([self.P1array[ElemInd-1], self.P2array[
                ElemInd-1]])
            ppp = ppp[(ppp > 0)]
            pp = ppp[(self.fluid[ppp] == 1) & ~(self.trappedNW[ppp])]
            
            # update Pc and the filling list
            assert pp.size > 0
            [*map(lambda i: self.do.isTrapped(i, 1, self.trappedNW), pp)]
            self.__computePc__(self.capPresMin, pp)
              
        except AssertionError:
            pass

        try:
            assert k <= self.nPores
            ElemInd = k
            assert not self.do.isTrapped(k, 1, self.trappedNW)
            #print(k, capPres, self.capPresMin)
            self.fluid[k] = 0
            self.fillmech[k] = 1*(self.PistonPcAdv[k] == capPres) + 2*(
                self.porebodyPc[k] == capPres) + 3*(
                    self.snapoffPc[k] == capPres)
            self.cntP += 1
            self.invInsideBox += self.isinsideBox[k]

            tt = self.PTConData[ElemInd]+self.nPores
            tt = tt[(self.fluid[tt] == 1) & ~(self.trappedNW[tt])]

            # update Pc and the filling list
            assert tt.size > 0
            [*map(lambda i: self.do.isTrapped(i, 1, self.trappedNW), tt)]
            self.__computePc__(self.capPresMin, tt)

        except AssertionError:
            pass

    def __CondTPImbibition__(self):
        # to suppress the FutureWarning and SettingWithCopyWarning respectively
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pd.options.mode.chained_assignment = None

        arrr = np.ones(self.totElements, dtype='bool')
        arrrS = arrr[self.elemSquare]
        arrrT = arrr[self.elemTriangle]
        arrrC = arrr[self.elemCircle]
        Pc = np.ones(self.totElements)*self.capPresMin

        try:
            assert (arrrS.sum() > 0)
            curConAng = self.contactAng.copy()
            apexDist = np.empty_like(self.hingAngSq.T)
            conAngPS, apexDistPS = self.do.cornerApex(
                self.elemSquare, arrrS, self.halfAnglesSq[:, np.newaxis], Pc[self.elemSquare],
                curConAng, self.cornExistsSq.T, self.initOrMaxPcHistSq.T,
                self.initOrMinApexDistHistSq.T, self.advPcSq.T,
                self.recPcSq.T, apexDist, self.initedApexDistSq.T)
            
            self._cornArea[self.elemSquare[arrrS]], self._cornCond[
                self.elemSquare[arrrS]] = self.do.calcAreaW(
                arrrS, self.halfAnglesSq, conAngPS, self.cornExistsSq, apexDistPS)
        except AssertionError:
            pass

        try:
            assert (arrrT.sum() > 0)
            curConAng = self.contactAng.copy()
            apexDist = np.empty_like(self.hingAngTr.T)
            conAngPT, apexDistPT = self.do.cornerApex(
                self.elemTriangle, arrrT, self.halfAnglesTr.T, Pc[self.elemTriangle],
                curConAng, self.cornExistsTr.T, self.initOrMaxPcHistTr.T,
                self.initOrMinApexDistHistTr.T, self.advPcTr.T,
                self.recPcTr.T, apexDist, self.initedApexDistTr.T)
            
            self._cornArea[self.elemTriangle[arrrT]], self._cornCond[
                self.elemTriangle[arrrT]] = self.do.calcAreaW(
                arrrT, self.halfAnglesTr, conAngPT, self.cornExistsTr, apexDistPT)
        except AssertionError:
            pass

        try:
            assert (arrrC.sum() > 0)
            self._cornArea[self.elemCircle[arrrC]] = 0.0
            self._cornCond[self.elemCircle[arrrC]] = 0.0
        except  AssertionError:
            pass
        
        try:
            assert (self._cornArea[arrr] <= self.AreaSPhase[arrr]).all()
        except AssertionError:
            cond = (self._cornArea[arrr] > self.AreaSPhase[arrr])
            self._cornArea[arrr & cond] = self.AreaSPhase[arrr & cond]
        try:
            assert (self._cornCond[arrr] <= self.gwSPhase[arrr]).all()
        except AssertionError:
            cond = (self._cornCond[arrr] > self.gwSPhase[arrr])
            self._cornCond[arrr & cond] = self.gwSPhase[arrr & cond]
        
        self._centerArea[arrr] = self.AreaSPhase[arrr] - self._cornArea[arrr]
        self._centerCond[arrr] = np.where(self.AreaSPhase[arrr] != 0.0, self._centerArea[
                arrr]/self.AreaSPhase[arrr]*self.gnwSPhase[arrr], 0.0)
        
        self.__updateAreaCond__()
        

    def __updateAreaCond__(self):
        arrr = (~self.trappedNW)

        try:
            cond2 = arrr & (self.fluid == 0)
            assert cond2.sum() > 0
            self._areaWP[cond2] = self.AreaSPhase[cond2]
            self._areaNWP[cond2] = 0.0
            self._condWP[cond2] = self.gwSPhase[cond2]
            self._condNWP[cond2] = 0.0
        except AssertionError:
            pass

        try:
            cond1 = arrr & (self.fluid == 1) & (self.Garray <= self.bndG2)
            assert cond1.sum() > 0
            self._areaWP[cond1] = np.maximum(np.minimum(
                self._cornArea[cond1], self.AreaSPhase[cond1]), 0.0)
            self._areaNWP[cond1] = np.maximum(np.minimum(
                self._centerArea[cond1], self.AreaSPhase[cond1]), 0.0)
            self._condWP[cond1] = np.maximum(np.minimum(
                self._cornCond[cond1], self.gwSPhase[cond1]), 0.0)
            self._condNWP[cond1] = np.maximum(np.minimum(
                self._centerCond[cond1], self.gnwSPhase[cond1]), 0.0)
        except AssertionError:
            pass

        try:
            cond3 = arrr & (self.fluid == 1) & (self.Garray > self.bndG2)
            assert cond3.sum() > 0
            self._areaWP[cond3] = 0.0
            self._areaNWP[cond3] = self.AreaSPhase[cond3]
            self._condWP[cond3] = 0.0
            self._condNWP[cond3] = self.gnwSPhase[cond3]
        except AssertionError:
            pass
        
    
    def __initCornerApex__(self):
        arrr = ((self.fluid == 1) & (~self.trappedW))
        arrrS = arrr[self.elemSquare]
        arrrT = arrr[self.elemTriangle]
        
        self.do.initCornerApex(
            self.elemTriangle, arrrT, self.halfAnglesTr, self.cornExistsTr, self.initedTr,
            self.recPcTr, self.advPcTr, self.initedApexDistTr, self.trappedW)
    
        self.do.initCornerApex(
            self.elemSquare, arrrS, self.halfAnglesSq, self.cornExistsSq, self.initedSq,
            self.recPcSq, self.advPcSq, self.initedApexDistSq, self.trappedW)
        
    
    def __computePistonPc__(self):
        conda = (self.fluid == 0)
        condb = (self.fluid == 1) & (self.Garray < self.bndG2)  #polygons filled with w
        condc = (self.fluid == 1) & (self.Garray >= self.bndG2) #circles filled with nw

        self.PistonPcAdv[conda | condc] = 2.0*self.sigma*np.cos(
             self.thetaAdvAng[conda | condc])/self.Rarray[conda | condc]
        conda = conda & (self.maxPc < self.PistonPcRec)
        self.PistonPcAdv[conda] = self.maxPc*np.cos(self.thetaAdvAng[conda])/np.cos(
            self.thetaRecAng[conda])
        
        normThresPress = (self.Rarray*self.maxPc)/self.sigma
        angSum = np.zeros(self.totElements)
        angSum[self.elemTriangle] = np.cos(self.thetaRecAng[
            self.elemTriangle, np.newaxis] + self.halfAnglesTr).sum(axis=1)
        angSum[self.elemSquare] = np.cos(self.thetaRecAng[
            self.elemSquare, np.newaxis] + self.halfAnglesSq).sum(axis=1)
        rhsMaxAdvConAng = (-4.0*self.Garray*angSum)/(
            normThresPress-np.cos(self.thetaRecAng)+12.0*self.Garray*np.sin(self.thetaRecAng))
        rhsMaxAdvConAng = np.minimum(1.0, np.maximum(rhsMaxAdvConAng, -1.0))
        m_maxConAngSpont = np.arccos(rhsMaxAdvConAng)

        condd = condb & (self.thetaAdvAng < m_maxConAngSpont) #calculte PHing
        self.__PistonPcHing__(condd)

        conde = np.zeros(self.totElements, dtype='bool')
        conde[self.elemTriangle] = condb[self.elemTriangle] & (~condd[self.elemTriangle]) & (
            self.thetaAdvAng[self.elemTriangle] <= np.pi/2+self.halfAnglesTr[:, 0])
        self.PistonPcAdv[conde] = 2.0*self.sigma*np.cos(
             self.thetaAdvAng[conde])/self.Rarray[conde]
        
        condf = condb & (~condd) & (~conde)         #use Imbnww
        #self.__PistonPcImbnww__(condf) this should be run for condf later
        self.PistonPcAdv[condf] = 2.0*self.sigma*np.cos(
             self.thetaAdvAng[condf])/self.Rarray[condf]
        

    def __PistonPcHing__(self, arrr):
        # compute entry capillary pressures for piston displacement

        arrrS = arrr[self.elemSquare]
        arrrT = arrr[self.elemTriangle]
        
        try:
            assert arrrT.sum() > 0
            self.PistonPcAdv[self.elemTriangle[arrrT]] = self.Pc_pistonHing(
                self.elemTriangle, arrrT, self.halfAnglesTr.T, self.cornExistsTr,
                self.initOrMaxPcHistTr, self.initOrMinApexDistHistTr, self.advPcTr,
                self.recPcTr, self.initedApexDistTr)
        except AssertionError:
            pass
        try:
            assert arrrS.sum() > 0
            self.PistonPcAdv[self.elemSquare[arrrS]] = self.Pc_pistonHing(
                self.elemSquare, arrrS, self.halfAnglesSq, self.cornExistsSq,
                self.initOrMaxPcHistSq, self.initOrMinApexDistHistSq, self.advPcSq,
                self.recPcSq, self.initedApexDistSq)
            #from IPython import embed; embed()
        except AssertionError:
            pass
       

    def Pc_pistonHing(self, arr, arrr, halfAng, m_exists, m_initOrMaxPcHist,
                      m_initOrMinApexDistHist, advPc, recPc, initedApexDist):
        
        newPc = 1.1*self.sigma*2.0*np.cos(self.thetaAdvAng[arr])/self.Rarray[arr]
        
        arrr1 = arrr.copy()
        apexDist = np.zeros(arrr.size)
        counter = 0
        while True:
            oldPc = newPc.copy()
            sumOne, sumTwo = np.zeros(arrr.size), np.zeros(arrr.size)
            sumThree, sumFour = np.zeros(arrr.size), np.zeros(arrr.size)
            for i in range(m_exists.shape[1]):
                cond1 = arrr1 & m_exists[:, i]                
                conAng, apexDist = self.do.cornerApex(
                    arr, cond1, halfAng[i], oldPc, self.thetaAdvAng.copy(), m_exists[:, i], m_initOrMaxPcHist[:, i], m_initOrMinApexDistHist[:, i], advPc[:, i],
                    recPc[:, i], apexDist, initedApexDist[:, i], accurat=True, overidetrapping=True)

                partus = (apexDist*np.sin(halfAng[i])*oldPc/self.sigma)

                try:
                    assert (abs(partus[cond1]) <= 1.0).all()
                except AssertionError:
                    print("partus has entries outside the range!", cond1.sum(), counter)
                    from IPython import embed; embed()              

                sumOne[cond1] += (apexDist*np.cos(conAng))[cond1]
                sumTwo[cond1] += (np.pi/2-conAng-halfAng[i])[cond1]
                sumThree[cond1] += (np.arcsin(partus[cond1]))
                sumFour[cond1] += apexDist[cond1]

            a = (2*sumThree-sumTwo)
            b = ((np.cos(self.thetaAdvAng[arr])*(self.Rarray[arr]/(
                2*self.Garray[arr]))) -2*sumFour +sumOne)
            c = (-pow(self.Rarray[arr], 2)/(4*self.Garray[arr]))

            arr1 = pow(b, 2)-np.array(4*a*c)
            cond = (arr1 > 0)
            newPc[arrr1] = (self.sigma*(2*a[arrr1])/(
                (-b+np.sqrt(arr1))*cond + (-b)*(~cond))[arrr1])
            err = 2.0*abs((newPc - oldPc)/(abs(oldPc)+abs(newPc)+1.0e-3))[arrr1]
            counter += 1
            try:
                assert (err < self.EPSILON).all() or (counter > self.MAX_ITER)
                break
            except AssertionError:
                arrr1[arrr1] = (err >= self.EPSILON)


        newPc[np.isnan(newPc)] = 0.0

        return newPc[arrr]
    

    def __PistonPcImbnww__(self, arrr, halfAng):
        try:
            assert arrr.sum() == 0
        except AssertionError:
            conAng = math.pi - self.thetaAdvAng[arrr]
            potentialCurveRad = []
            sOne = np.zeros(arrr.sum())
            sTwo = np.zeros(arrr.sum())
            sThree = np.zeros(arrr.sum())
            for i in range(halfAng.shape[0]):
                cond = (self.thetaAdvAng[arrr] > math.pi/2.0 + halfAng[i])
                sOne[cond] += (np.cos(conAng) * np.cos(conAng + halfAng[i]) /
                        np.sin(halfAng[i]) - (np.pi/2.0 - conAng - halfAng[i]))
                sTwo[cond] += np.cos(conAng + halfAng[i]) / np.sin(halfAng[i])
                sThree[cond] += 2.0 * (math.pi/2.0 - conAng - halfAng[i])

                dFact = sOne - 2.0 * sTwo * np.cos(conAng) + sThree
                rootFact = 1.0 + 4.0*self.shapeFactor*dFact/(np.cos(conAng)*np.cos(conAng))

                radOne = self.Rarray[arrr]*np.cos(conAng)*(1.0 - np.sqrt(rootFact))/(
                    4.0 * self.Garray[arrr] * dFact)
                radTwo = (self.Rarray[arrr]*np.cos(conAng)*(1.0 + np.sqrt(rootFact))/(
                    4.0*self.shapeFactor*dFact))
                potentialCurveRad.append(np.maximum(radOne, radTwo))
            

        

            # Create an array of tension values
            tension = self.comn.oil().interfacialTen() * np.ones(self.numCorners)
            
            # Calculate pc values
            pc = tension / potentialCurveRad
            
            # Initialize the layerPc array with negative infinity values
            layerPc = np.full(self.numCorners, float('-inf'))
            
            # Calculate layerPc values for corners where the waterInCorner exists
            cond2 = self.waterInCorner.cornerExists()
            layerPc[cond2] = self.oilLayer[cond2].entryPc()

            # Find the first corner with pc > layerPc and return the corresponding result
            cond3 = pc > layerPc
            first_true_index = np.argmax(cond3)
            
            if any(cond3):
                return self.comn.oil().interfacialTen() / potentialCurveRad[first_true_index]

            # If no corner meets the condition, return the default value
            return self.comn.oil().interfacialTen() * (2.0 * np.cos(self.thetaAdvAng)) / self.R


    
    def __computeSnapoffPc__(self):
        # update entry capillary pressure for Snap-off filling
        arrr = (self.fluid == 1)
        arrrTr = arrr[self.elemTriangle]
        arrTr = self.elemTriangle[arrrTr]
        arrrSq = arrr[self.elemSquare]
        arrSq = self.elemSquare[arrrSq]
        self.snapoffPc[arrr & (self.trappedNW)] = -1e28
        
        try:
            assert arrrTr.sum() > 0
            cond = arrrTr & (~self.trappedNW[self.elemTriangle]) & (self.thetaAdvAng[
                self.elemTriangle] < (np.pi/2.0 - self.halfAnglesTr[:, 0]))
            arr = self.elemTriangle[cond]
            snapoffPc1 = self.sigma*np.cos(self.thetaAdvAng[
                self.elemTriangle]+self.halfAnglesTr[:, 0])/((
                self.Rarray[self.elemTriangle]/np.tan(self.halfAnglesTr[:,0]) +
                self.Rarray[self.elemTriangle]/np.tan(self.halfAnglesTr[:,2]) -
                self.initedApexDistTr[:,2])*np.sin(self.halfAnglesTr[:,0]))
            oldPc = np.full(cond.size, self.maxPc)
            arrtrapped = self.elemTriangle[self.trappedNW[self.elemTriangle]]
            oldPc[self.trappedNW[self.elemTriangle]] = self.PcI[arrtrapped]
            L0pL2 = (self.Rarray[self.elemTriangle]/np.tan(self.halfAnglesTr[:,0]) +
                     self.Rarray[self.elemTriangle]/np.tan(self.halfAnglesTr[:,1]))
            condc = cond.copy()
            
            i=0
            while True:
                apexDist = np.zeros(condc.size)
                teta2 = self.thetaAdvAng.copy()
                conAng, apexDist = self.do.cornerApex(
                        self.elemTriangle, condc, self.halfAnglesTr[:, 1], oldPc, teta2,
                        self.cornExistsTr[:, 1], self.initOrMaxPcHistTr[:, 1],
                        self.initOrMinApexDistHistTr[:, 1], self.advPcTr[:, 1],
                        self.recPcTr[:, 1], apexDist, self.initedApexDistTr[:, 1],
                        accurat=True, overidetrapping=True)

                ang=conAng[condc]
                angadv=teta2[self.elemTriangle[condc]]
                rL2 = -apexDist[condc]*np.sin(self.halfAnglesTr[condc,1])/(
                    self.sigma*np.sin(ang+self.halfAnglesTr[condc,1]))
                func = oldPc[condc] - self.sigma*(
                    np.cos(angadv)/np.tan(self.halfAnglesTr[condc,0]) - np.sin(angadv) + np.cos(ang)/np.tan(
                        self.halfAnglesTr[condc,1]) - np.sin(ang)) / L0pL2[condc]
                funcDpc = 1 + self.sigma*(rL2*np.sin(ang)/np.tan(
                    self.halfAnglesTr[condc,1]) +  rL2*np.cos(ang)) / L0pL2[condc]
                condd = (abs(funcDpc)>1.0e-32)
                

                newPc = oldPc[condc]
                newPc[condd] = newPc[condd] - func[condd]/funcDpc[condd]
                newPc[~condd] = newPc[~condd] - func[~condd]
                try:
                    assert i > self.MAX_ITER/2
                    newPc = 0.5*(newPc+oldPc[condc])
                except AssertionError:
                    pass
    
                err = abs(newPc-oldPc[condc])/(abs(oldPc[condc])+1e-8)
                conde = (err < self.EPSILON) & (ang <= angadv+1e-6)
                condf = (err < self.EPSILON) & ~(ang <= angadv+1e-6)
                newPc[conde] = np.maximum(newPc[conde], snapoffPc1[condc][conde])+1e-4
                newPc[condf] = snapoffPc1[condc][condf]+1e-4
                oldPc[condc] = newPc
                condc[condc] = (err >= self.EPSILON)

                try:
                    assert condc.sum() == 0
                    self.snapoffPc[arr] = oldPc[cond]
                    break            
                except AssertionError:
                    i += 1
        except AssertionError:
            pass

        try:
            assert arrrSq.sum() > 0
            cond = arrrSq & (~self.trappedNW[self.elemSquare]) & (self.thetaAdvAng[
                self.elemSquare] < (np.pi/2.0 - self.halfAnglesSq[0]))
            arr = self.elemSquare[cond]
            self.snapoffPc[arr] = self.sigma/self.Rarray[arr]*(
                np.cos(self.thetaAdvAng[arr]) - np.sin(self.thetaAdvAng[arr]))            
        except AssertionError:
            pass
        

    def __computeSnapoffPc1__(self, Pc: float):
        # update entry capillary pressure for Snap-off filling
        arrr = (self.fluid == 1)
        arrrTr = arrr[self.elemTriangle]
        arrTr = self.elemTriangle[arrrTr]
        arrSq = self.elemSquare[arrr[self.elemSquare]]
        
        cotBetaTr = 1/np.tan(self.halfAnglesTr)
        cotBetaSq = 1/np.tan(self.halfAnglesSq)
        
        snapoffPc1 = self.sigma/self.Rarray[arrTr]*(np.cos(
            self.thetaAdvAng[arrTr]) - 2*np.sin(
            self.thetaAdvAng[arrTr])/cotBetaTr[arrrTr, 0:2].sum(axis=1))
        apexDistTr = self.sigma/self.maxPc*np.cos(self.thetaRecAng[
            arrTr][:, np.newaxis]+self.halfAnglesTr[arrrTr])/np.sin(
                self.halfAnglesTr[arrrTr])
        thetaHi = np.arccos(apexDistTr*np.sin(self.halfAnglesTr[arrrTr])/(
            self.sigma/Pc))
        snapoffPc2 = self.sigma/self.Rarray[arrTr] * (np.cos(
            self.thetaAdvAng[arrTr])*cotBetaTr[arrrTr, 0] - np.sin(
            self.thetaAdvAng[arrTr]) + np.cos(thetaHi[:, 2])*cotBetaTr[arrrTr, 2]
            - np.sin(thetaHi[:, 2]))/(cotBetaTr[:, [0, 2]][arrrTr].sum(axis=1))
        self.snapoffPc[arrTr] = np.max([snapoffPc1, snapoffPc2], axis=0)
        self.snapoffPc[arrSq] = self.sigma/self.Rarray[arrSq]*(np.cos(
            self.thetaAdvAng[arrSq]) - 2*np.sin(self.thetaAdvAng[arrSq])/(
            cotBetaSq[0]*2))
         

    def __func1(self, i):
        try:
            return (self.fluid[self.PTConData[i]+self.nPores] == 0).sum()
        except IndexError:
            return 0.0
        
    def __func2(self, i):
        try:
            arr=self.PTConData[i][(self.fluid[self.PTConData[i]+self.nPores] == 0)]+self.nPores
            return self.PistonPcAdv[arr].max()
        except (IndexError, ValueError):
            return 0.0
        
    def __func3(self, i):
        try:
            arr=np.array([self.P1array[i-1], self.P2array[i-1]])
            return self.PistonPcAdv[arr[self.fluid[arr]==0]].max()
        except (IndexError, ValueError):
            return 0.0
        
    def __removeElem(self, i):
        try:
            self.ElemToFill.remove(i)
        except ValueError:
            try:
                m = npi.indices(self.ElemToFill, [i])[0]
                del self.ElemToFill[m]
            except KeyError:
                pass

    def LookupList(self, k):
        return (-round(self.PcI[k], 9), k <= self.nPores, -k)

    def __computePc__(self, Pc, arr, update=True):
        cond = (self.fluid == 1)
        arrP = arr[arr <= self.nPores]
        arrT = arr[arr > self.nPores]
        arrrCP = arrP[cond[arrP]]     # pores filled with nw
        arr1 = np.array([*map(lambda i: self.__func1(i), arrrCP)])
        cond1 = (arr1 > 0) & (self.thetaAdvAng[arrrCP] < np.pi/2.0) #pores for porebody filling
        
        entryPc = self.PistonPcAdv.copy()
        self.__porebodyFilling__(arrrCP[cond1])
        entryPc[arrrCP[cond1]] = self.porebodyPc[arrrCP[cond1]]
        
        maxNeiPistonPrs = np.zeros(self.totElements)
        maxNeiPistonPrs[arrrCP] = np.array([*map(lambda i: self.__func2(i), arrrCP)])
        arrrCT = arrT[cond[arrT]]
        maxNeiPistonPrs[arrrCT] = np.array([*map(lambda i: self.__func3(i), arrrCT-self.nPores)])
        condb = (maxNeiPistonPrs > 0.0)
        entryPc[condb] = np.minimum(0.999*maxNeiPistonPrs[
            condb]+0.001*entryPc[condb], entryPc[condb])
        
        # Snap-off filling
        self.__computeSnapoffPc1__(Pc)
        #self.__computeSnapoffPc__()

        conda = (maxNeiPistonPrs > 0.0) & (entryPc > self.snapoffPc)
        entryPc[~conda&(self.Garray<self.bndG2)] = self.snapoffPc[~conda&(self.Garray<self.bndG2)]

        try:
            assert update
            
            diff = (self.PcI[arr] != entryPc[arr])
            [*map(lambda i: self.__removeElem(i), arr[diff])]
            self.PcI[arr[diff]] = entryPc[arr[diff]]
            
            # update the filling list
            self.ElemToFill.update(arr[diff]) # try .__add__(other)
        except AssertionError:
            self.PcI[arr] = entryPc[arr]
            self.ElemToFill.update(arr[(self.fluid[arr]==1)])
            


    def __porebodyFilling__(self, ind):
        try:
            assert ind.size > 0
        except AssertionError:
            return
        
        arr1 = np.array([self.PTConData[i][self.fluid[
                self.PTConData[i]+self.nPores] == 1].size for i in ind])
        arr1[arr1 > 5] = 5
        cond = (arr1 > 1)
        
        # Oren - not correct though!
        #sumrand = np.array([*map(lambda i: self.rand(i-1).sum(), arr1[cond])])*self.Rarray[
         #   ind[cond]]
        # Blunt2
        sumrand = np.zeros(ind.size)
        sumrand[cond] = np.array([*map(lambda i: (self.rand(i-1)*15000).sum(), arr1[cond])])
        try:
            # Blunt2
            Pc = self.sigma*(2*np.cos(self.thetaAdvAng[ind])/self.Rarray[
                ind] - sumrand)
        except TypeError:
            pass
        
        self.porebodyPc[ind] = Pc


    def __writeHeadersI__(self):
        result_dir = "./results_csv/"   
        self.file_name = os.path.join(result_dir, "FlowmodelOOP_"+
                            self.title+"_Imbibition_"+str(self._num)+".csv")

        self.resultI_str="======================================================================\n"
        self.resultI_str+="# Fluid properties:\nsigma (mN/m)  \tmu_w (cP)  \tmu_nw (cP)\n"
        self.resultI_str+="# \t%.6g\t\t%.6g\t\t%.6g" % (
            self.sigma*1000, self.muw*1000, self.munw*1000, )
        self.resultI_str+="\n# calcBox: \t %.6g \t %.6g" % (
            self.calcBox[0], self.calcBox[1], )
        self.resultI_str+="\n# Wettability:"
        self.resultI_str+="\n# model \tmintheta \tmaxtheta \tdelta \teta \tdistmodel"
        self.resultI_str+="\n# %.6g\t\t%.6g\t\t%.6g\t\t%.6g\t\t%.6g" % (
            self.wettClass, round(self.minthetai*180/np.pi,3), round(self.maxthetai*180/np.pi,3), self.delta, self.eta,)
        self.resultI_str+=self.distModel
        self.resultI_str+="\nmintheta \tmaxtheta \tmean  \tstd"
        self.resultI_str+="\n# %3.6g\t\t%3.6g\t\t%3.6g\t\t%3.6g" % (
            round(self.contactAng.min()*180/np.pi,3), round(self.contactAng.max()*180/np.pi,3), round(self.contactAng.mean()*180/np.pi,3), round(self.contactAng.std()*180/np.pi,3))
        
        self.resultI_str+="\nPorosity:  %3.6g" % (self.porosity)
        self.resultI_str+="\nMaximum pore connection:  %3.6g" % (self.maxPoreCon)
        self.resultI_str+="\nAverage pore-to-pore distance:  %3.6g" % (self.avgP2Pdist)
        self.resultI_str+="\nMean pore radius:  %3.6g" % (self.Rarray[self.poreList].mean())
        self.resultI_str+="\nAbsolute permeability:  %3.6g" % (self.absPerm)
        
        self.resultI_str+="\n======================================================================"
        self.resultI_str+="\n# Sw\t qW(m3/s)\t krw\t qNW(m3/s)\t krnw\t Pc\t Invasions"

        
        
        
        


