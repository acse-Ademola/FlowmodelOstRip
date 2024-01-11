from datetime import date
import sys
import os
import pandas as pd

sys.path.append("../Flowmodel")
from Flowmodel.inputData import InputData
from Flowmodel.network import Network
from Flowmodel.sPhase import SinglePhase
from Hysteresis import PDrainage, PImbibition, SecDrainage, SecImbibition
from plot import makePlot


# __DATE__ = "Jul 25 , 2023"
__DATE__ = date.today().strftime("%b") + " " + str(date.today().day) + ", " +\
      str(date.today().year)


def main():
    try:
        input_file_name = ""

        print("\nNetwork Model Code version 2 alpha, built: ", __DATE__, "\n")

        if len(sys.argv) > 1:
            input_file_name = sys.argv[1]
        else:
            input_file_name = input("Please input data file : ")

        input_data = InputData(input_file_name)
        netsim = Network(input_file_name)

        # Single Phase computation
        netsim = SinglePhase(netsim)
        netsim.singlephase()

        toPlot = True
        compWithLitData = True
        compWithPrevData = True
        drainPlot = False
        imbibePlot = False
        probablePlot = False
        writeData = True
        includeTrapping = False

        primaryD, primaryI = True, True

        # two Phase simulations
        if input_data.satControl():
            netsim.prop_drainage = {}
            netsim.prop_imbibition = {}
            for cycle in range(len(input_data.satControl())):
                netsim.finalSat, Pc, netsim.dSw, netsim.minDeltaPc,\
                 netsim.deltaPcFraction, netsim.calcKr, netsim.calcI,\
                 netsim.InjectFromLeft, netsim.InjectFromRight,\
                 netsim.EscapeFromLeft, netsim.EscapeFromRight =\
                 input_data.satControl()[cycle]
                netsim.filling = True

                if netsim.finalSat < netsim.satW:
                    # Drainage process
                    netsim.is_oil_inj = True
                    netsim.maxPc = Pc
                    if primaryD:
                        (netsim.wettClass, netsim.minthetai, netsim.maxthetai, netsim.delta,
                            netsim.eta, netsim.distModel, netsim.sepAng) = input_data.initConAng(
                                'INIT_CONT_ANG')
                        netsim = PDrainage(netsim, includeTrapping=includeTrapping,
                                                  writeData=writeData)
                        primaryD = False
                        netsim.prop_drainage['contactAng'] = netsim.contactAng.copy()
                        netsim.prop_drainage['thetaRecAng'] = netsim.thetaRecAng.copy()
                        netsim.prop_drainage['thetaAdvAng'] = netsim.thetaAdvAng.copy()
                    else:
                        netsim = SecDrainage(netsim, includeTrapping=includeTrapping,
                                             writeData=writeData)
                    netsim.drainage()
                    
                    if drainPlot:
                        drainage_results = {}
                        drainage_results['model'] = pd.read_csv(
                            netsim.fQ.name, names=[
                            'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
                            sep=',', skiprows=18, index_col=False)
    
                else:
                    # Imbibition process
                    netsim.minPc = Pc
                    if primaryI:
                        (netsim.wettClass, netsim.minthetai, netsim.maxthetai, netsim.delta,
                            netsim.eta, netsim.distModel, netsim.sepAng) = input_data.initConAng(
                                'EQUIL_CON_ANG')
                        netsim = PImbibition(netsim, writeData=writeData,
                                             includeTrapping=includeTrapping)
                        primaryI = False
                        netsim.prop_imbibition['contactAng'] = netsim.contactAng.copy()
                        netsim.prop_imbibition['thetaRecAng'] = netsim.thetaRecAng.copy()
                        netsim.prop_imbibition['thetaAdvAng'] = netsim.thetaAdvAng.copy()
                    else:
                        netsim = SecImbibition(netsim, writeData=writeData,
                                               includeTrapping=includeTrapping)
                    netsim.imbibition()

                    if imbibePlot:
                        imbibition_results = {}
                        imbibition_results['model'] = pd.read_csv(
                            netsim.fQ.name, names=[
                            'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
                            sep=',', skiprows=18, index_col=False)
            
            #netsim = Hysteresis(netsim)
            
            if toPlot:
                if drainPlot:
                    mkD = makePlot(netsim._num, netsim.title, drainage_results,
                                   compWithLitData, compWithPrevData, True, False, include=None)
                    mkD.pcSw()
                    mkD.krSw()
                if imbibePlot:
                    mkI = makePlot(netsim._num, netsim.title, imbibition_results, 
                                   compWithLitData, compWithPrevData, False, True, include=None)
                    mkI.pcSw()
                    mkI.krSw()
                if probablePlot:
                    mkP = makePlot(netsim._num, netsim.title, imbibition_results, True, True, False, True, include=None)

                #from IPython import embed; embed()



        else:
            pass
    except Exception as exc:
        print("\n\n Exception on processing: \n", exc, "Aborting!\n")
        return 1
    except:
        from IPython import embed; embed()
        print("\n\n Unknown exception! Aborting!\n")
        return 1

    return 0





if __name__ == "__main__":
    sys.exit(main())


