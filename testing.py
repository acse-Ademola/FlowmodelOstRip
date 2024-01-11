import pandas as pd
from plot import makePlot


drainage_results = {}
drainage_results['model'] = pd.read_csv(
    './results_csv/FlowmodelOOP_Bentheimer_Drainage_010725.csv', names=[
    'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
    sep=',', skiprows=18, index_col=False)

imbibition_results = {}
imbibition_results['model'] = pd.read_csv(
    './results_csv/FlowmodelOOP_Bentheimer_Imbibition_010725.csv', names=[
    'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
    sep=',', skiprows=18, index_col=False)

num = 1
title = 'Bentheimer'
drain = False
imbibe = False
probable = True

if drain:
    mkD = makePlot(num, title, drainage_results, True, True, True, False, include=None)
    mkD.pcSw()
    mkD.krSw()
if imbibe:
    mkI = makePlot(num, title, imbibition_results, True, True, False, True, include=None)
    mkI.pcSw()
    mkI.krSw()