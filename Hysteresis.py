import os
import numpy as np
from sortedcontainers import SortedList
import warnings
import pandas as pd
from itertools import chain

from TPhaseD import TwoPhaseDrainage
from Flowmodel.TPhaseImb import TwoPhaseImbibition
from Flowmodel.utilities import Computations


class PDrainage(TwoPhaseDrainage):
    def __new__(cls, obj, writeData=False, includeTrapping=True):
        obj.__class__ = PDrainage
        return obj
    
    def __init__(self, obj, writeData=False, includeTrapping=True):
        super().__init__(obj, writeData=writeData)
        self.includeTrapping = includeTrapping

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
            
            if not self.includeTrapping: self.identifyTrappedElements()
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
        
        if not self.includeTrapping: self.identifyTrappedElements()
        self.__CondTP_Drainage__()
        self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
        self.do.computePerm()
        self.resultD_str = self.do.writeResult(self.resultD_str, self.capPresMax)


    def popUpdateOilInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcD[k]
        self.capPresMax = np.max([self.capPresMax, capPres])

        try:
            assert k > self.nPores
            ElemInd = k-self.nPores
            if self.includeTrapping:
                assert not self.do.isTrapped(k, 0, self.trappedW)
                self.fluid[k] = 1
                self.PistonPcRec[k] = self.centreEPOilInj[k]
                ppp = np.array([self.P1array[ElemInd-1], self.P2array[
                    ElemInd-1]])
                pp = ppp[(self.fluid[ppp] == 0) & ~(self.trappedW[ppp])]
                [*map(lambda i: self.do.isTrapped(i, 0, self.trappedW), pp)]

                self.cntT += 1
                self.invInsideBox += self.isinsideBox[k]
                self.__update_PcD_ToFill__(pp)
            else:
                self.fluid[k] = 1
                self.PistonPcRec[k] = self.centreEPOilInj[k]
                ppp = np.array([self.P1array[ElemInd-1], self.P2array[
                    ElemInd-1]])
                pp = ppp[(self.fluid[ppp] == 0)]

                self.cntT += 1
                self.invInsideBox += self.isinsideBox[k]
                self.__update_PcD_ToFill__(pp)
        except AssertionError:
            pass
        except IndexError:
            pass

        try:
            assert k <= self.nPores
            if self.includeTrapping:
                assert not self.do.isTrapped(k, 0, self.trappedW)
                self.fluid[k] = 1
                self.PistonPcRec[k] = self.centreEPOilInj[k]
                thr = self.PTConData[k]+self.nPores
                thr = thr[(self.fluid[thr] == 0) & ~(self.trappedW[thr])]
                [*map(lambda i: self.do.isTrapped(i, 0, self.trappedW), thr)]

                self.cntP += 1
                self.invInsideBox += self.isinsideBox[k]
                self.__update_PcD_ToFill__(thr)
            
            else:
                self.fluid[k] = 1
                self.PistonPcRec[k] = self.centreEPOilInj[k]
                thr = self.PTConData[k]+self.nPores
                thr = thr[(self.fluid[thr] == 0)]

                self.cntP += 1
                self.invInsideBox += self.isinsideBox[k]
                self.__update_PcD_ToFill__(thr)

        except AssertionError:
            pass
        except IndexError:
            pass

    def identifyTrappedElements(self, fluid):
        Notdone = (self.fluid==1)
        tin = list(self.conTToIn[Notdone[self.conTToIn+self.nPores]])
        tout = self.conTToOut[Notdone[self.conTToOut+self.nPores]]
        self.trappedNW[:] = True
        conn = np.zeros(self.totElements, dtype='bool')

        while True:
            try:
                conn[:] = False
                tt = tin.pop(0)
                Notdone[tt+self.nPores] = False
                conn[tt+self.nPores] = True
                while True:
                    try:
                        pp = np.array([self.P1array[tt-1], self.P2array[tt-1]])
                        pp = pp[Notdone[pp]]
                        Notdone[pp] = False
                        conn[pp] = True

                        tt = np.array([*chain(*self.PTConData[pp])])
                        tt = tt[Notdone[tt+self.nPores]]
                        Notdone[tt+self.nPores] = False
                        conn[tt+self.nPores] = True
                    except IndexError:
                        try:
                            tin = np.array(tin)
                            tin = list(tin[Notdone[tin]])
                        except IndexError:
                            tin=[]
                        break
                if any(conn[tout]):
                    self.trappedNW[conn] = False
                
            except IndexError:
                break

    def __writeHeaders__(self):
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


    def findNum(self):
        self._num = 1
        label = 'wt' if self.includeTrapping else 'nt'
        result_dir = "./results_csv/"
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        while True:         
            file_name = os.path.join(result_dir, "Flowmodel_"+
                                self.title+"_Drainage_cycle1_"+label+"_"+str(self._num)+".csv")
            if os.path.isfile(file_name): self._num += 1
            else:
                break
        self.file_name = file_name
        
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    

class PImbibition(TwoPhaseImbibition):
    def __new__(cls, obj, writeData=False, includeTrapping=True):
        obj.__class__ = PImbibition
        return obj
    
    def __init__(self, obj, writeData=False, includeTrapping=True):
        super().__init__(obj, writeData=writeData)
        self.writeData = writeData
        self.includeTrapping = includeTrapping

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
                    try:
                        assert self.satW >= 0.5
                        if not self.ensureConnectivity():
                            self.filling = False
                            break
                    except AssertionError:
                        pass
                        
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

            if not self.filling:
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


    def ensureConnectivity(self):
        self.gNWPhase[self.fluid==0]=0.0
        gnwL = self.do.computegL(self.gNWPhase)
        qNW = self.do.computeFlowrate(gnwL)
        return (qNW!=0.0)
    

    def popUpdateWaterInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcI[k]
        self.capPresMin = np.min([self.capPresMin, capPres])

        try:
            assert k > self.nPores
            ElemInd = k-self.nPores
            if self.includeTrapping:
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

            else:
                self.fluid[k] = 0
                self.fillmech[k] = 1*(self.PistonPcAdv[k] == capPres)+3*(
                    self.snapoffPc[k] == capPres)
                self.cntT += 1
                self.invInsideBox += self.isinsideBox[k]

                ppp = np.array([self.P1array[ElemInd-1], self.P2array[
                    ElemInd-1]])
                pp = ppp[(ppp > 0) & (self.fluid[ppp] == 1)]
                
                # update Pc and the filling list
                assert pp.size > 0
                self.__computePc__(self.capPresMin, pp)
            
    
        except AssertionError:
            pass

        try:
            assert k <= self.nPores
            ElemInd = k
            if self.includeTrapping:
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

            else:
                self.fluid[k] = 0
                self.fillmech[k] = 1*(self.PistonPcAdv[k] == capPres) + 2*(
                    self.porebodyPc[k] == capPres) + 3*(
                        self.snapoffPc[k] == capPres)
                self.cntP += 1
                self.invInsideBox += self.isinsideBox[k]

                tt = self.PTConData[ElemInd]+self.nPores
                tt = tt[(self.fluid[tt] == 1)]

                # update Pc and the filling list
                assert tt.size > 0
                self.__computePc__(self.capPresMin, tt)

        except AssertionError:
            pass









class SecImbibition(PImbibition):
    def __new__(cls, obj, writeData=False, includeTrapping=True):
        obj.__class__ = SecImbibition
        return obj
    
    def __init__(self, obj, writeData=False, includeTrapping=True):
        self.do = Computations(self)
    
        self.fluid[[-1, 0]] = 0, 1  
        
        self.is_oil_inj = False
        self._updateCornerApex_()
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        self.capPresMin = self.maxPc
        
        self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_imbibition.values()
        self.__initCornerApex__()
        self.__computePistonPc__()
        self.__computePc__(self.maxPc, self.elementLists, False)

        self._areaWP = self._cornArea.copy()
        self._areaNWP = self._centerArea.copy()
        self._condWP = self._cornCond.copy()
        self._condNWP = self._centerCond.copy()

        self.writeData = writeData
        if self.writeData: self.__writeHeadersI__()
        else: self.resultI_str = ""


    def __writeHeaders__(self):
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
            round(self.contactAng.min()*180/np.pi,3), round(self.contactAng.max()*180/np.pi,3), 
            round(self.contactAng.mean()*180/np.pi,3), round(self.contactAng.std()*180/np.pi,3))
        
        self.resultI_str+="\nPorosity:  %3.6g" % (self.porosity)
        self.resultI_str+="\nMaximum pore connection:  %3.6g" % (self.maxPoreCon)
        self.resultI_str+="\nAverage pore-to-pore distance:  %3.6g" % (self.avgP2Pdist)
        self.resultI_str+="\nMean pore radius:  %3.6g" % (self.Rarray[self.poreList].mean())
        self.resultI_str+="\nAbsolute permeability:  %3.6g" % (self.absPerm)
        
        self.resultI_str+="\n======================================================================"
        self.resultI_str+="\n# Sw\t qW(m3/s)\t krw\t qNW(m3/s)\t krnw\t Pc\t Invasions"  



class SecDrainage(TwoPhaseDrainage):
    def __new__(cls, obj, writeData=False, includeTrapping=True):
        obj.__class__ = SecDrainage
        return obj
    
    def __init__(self, obj, writeData=False, includeTrapping=True):
        self.do = Computations(self)
        #from IPython import embed; embed()
        self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_drainage.values()
        self.cornExistsTr[:] = False
        self.cornExistsSq[:] = False
        self.initedTr[:] = False
        self.initedSq[:] = False
        
        self.maxPc = self.capPresMax
        self.capPresMax = 0        
        
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        ElemToFill = self.nPores+self.conTToIn
        self.ElemToFill.update(ElemToFill)
        self.NinElemList = np.ones(self.totElements, dtype='bool')
        self.NinElemList[ElemToFill] = False

        self._cornArea = self.AreaWPhase.copy()
        self._centerArea = self.AreaNWPhase.copy()
        self._cornCond = self.gwSPhase.copy()
        self._centerCond = self.gnwSPhase.copy()

        self.writeData = writeData
        if self.writeData: self.__writeHeadersD__()
        else: self.resultD_str = ""