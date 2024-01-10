import pandas as pd
from plot import makePlot


drainage_results = {}
drainage_results['model'] = pd.read_csv(
    './results_csv/FlowmodelOOP_Bentheimer_Drainage_1.csv', names=[
    'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
    sep=',', skiprows=18, index_col=False)

imbibition_results = {}
imbibition_results['model'] = pd.read_csv(
    './results_csv/FlowmodelOOP_Bentheimer_Imbibition_1.csv', names=[
    'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
    sep=',', skiprows=18, index_col=False)

sec_drainage_results = {}
sec_drainage_results['model'] = pd.read_csv(
    './results_csv/FlowmodelOOP_Bentheimer_SDrainage_1.csv', names=[
    'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
    sep=',', skiprows=18, index_col=False)

num = 1
title = 'Bentheimer'
drain = False
imbibe = False
probable = True
hysteresis = True


if drain:
    mkD = makePlot(num, title, drainage_results, drain=True)
    print(mkD)
    mkD.pcSw()
    #mkD.krSw()
if imbibe:
    mkI = makePlot(num, title, imbibition_results, imbibe=True)
    mkI.pcSw()
    #mkI.krSw()
if hysteresis:
    results = {'drain': drainage_results['model'], 'imbibe': imbibition_results['model'],
               'sec_drain': sec_drainage_results['model']}
    mkH = makePlot(num, title, results, hysteresis=True)
    mkH.pcSw()