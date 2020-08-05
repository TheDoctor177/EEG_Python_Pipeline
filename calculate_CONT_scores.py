# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:13:48 2020

@author: ReneS
"""

import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import pyconscious as pc


# define path and filename (here you might want to loop over datasets!)
sName = {"ane_SD_EMG_1010", "ane_SD_EMG_1016", "ane_SD_EMG_1017", "ane_SD_EMG_1022", "ane_SD_EMG_1024", "ane_SD_EMG_1033", "ane_SD_EMG_1036", "ane_SD_EMG_1045", "ane_SD_EMG_1046", "ane_SD_EMG_1054"}
# sName = {"ane_SD_EMG_1045"}
sName = sorted(sName)
conditions = {"awake_rest_ec", "awake_rest_eo", "SED_1_rest", "SED_2_rest", "SED_3_rest"}
# conditions = {"_awake_rest_ec", "_SED_3_rest"}
# conditions = {"_SED_3_rest"}
ending = "_clean_epo.fif"
# conditions = {"SED_1_rest_2"}

result = {
    'SubID':sName,
    'awake_rest_ec': np.zeros(len(sName)),
    'awake_rest_eo': np.zeros(len(sName)),
    'SED_1_rest': np.zeros(len(sName)),
    'SED_2_rest': np.zeros(len(sName)),
    'SED_3_rest': np.zeros(len(sName))}
 
# resultSCEdf = pd.DataFrame(result, index = sName, columns=['awake_rest_ec','awake_rest_eo', 'SED_1_rest', 'SED_2_rest', 'SED_3_rest'])
# resultACEdf = pd.DataFrame(result, index = sName, columns=['awake_rest_ec','awake_rest_eo', 'SED_1_rest', 'SED_2_rest', 'SED_3_rest'])
resultLZCdf = pd.DataFrame(result, index = sName, columns=['awake_rest_ec','awake_rest_eo', 'SED_1_rest', 'SED_2_rest', 'SED_3_rest'])


for sub in sName:
    print("Now on Subject %s" % sub)
    for cond in conditions:
        # 0.1 Decide which participant and which condition
        filename = sub + '_' + cond
        fullfilename = sub + '_' + cond + ending
        #filepath = Path("C:/Users/imadjb/Documents/EEG_ANALYSIS/ane_SD_1016")
        # outpath = "E:/Anesthesia/EEG_preProcessed/" + sub + "/"
        filepath = ("D:/Anesthesia/EEG_preProcessedAuto/" + sub + "/" + fullfilename)
        # file = filepath / fullfilename
        # try: 
        #     os.mkdir(outpath)
        #     print('Path created')
        # except:
        #     print("Path exists")
        
        # 1. load data (fif file)
        try:
            data = mne.read_epochs(filepath)
            data.load_data()
            
            print("Calculating File %s" % filepath)
        except:
            print("**********************************************************")
            print("File %s does not exist" % filepath)
            print("**********************************************************")
            continue
        
        data.apply_baseline((None,None))
        # data.resample(250, npad = "auto")
        data = mne.preprocessing.compute_current_source_density(data)
        # # 15. Calculate LZC
        # finData = data.get_data(picks = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'])
        finData = data.get_data(picks = 'csd')
        resultLZ = pc.LZc(finData)
        resultLZCdf.loc[[sub],[cond]] = resultLZ

        # # The following need 3D arrays to function...
        # resultSCE = pc.SCE(finData)
        # resultACE = pc.ACE(finData)
        # resultSCEdf.loc[[sub],[cond]] = resultSCE
        # resultACEdf.loc[[sub],[cond]] = resultACE
        
resultLZCdf.to_excel("D:/Anesthesia/LZCresults_laplacian.xlsx")
# resultACEdf.to_excel("E:/Anesthesia/ACEresults_nineChan_sRate1000.xlsx")
# resultSCEdf.to_excel("E:/Anesthesia/SCEresults_nineChan.xlsx")