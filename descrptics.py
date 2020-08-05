# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:40:40 2020

@author: ReneS
"""

import mne
import pandas as pd
import numpy as np



# define path and filename (here you might want to loop over datasets!)
sName = {"ane_SD_EMG_1010", "ane_SD_EMG_1016", "ane_SD_EMG_1017", "ane_SD_EMG_1022", "ane_SD_EMG_1024", "ane_SD_EMG_1033", "ane_SD_EMG_1036", "ane_SD_EMG_1045", "ane_SD_EMG_1046", "ane_SD_EMG_1054"}
# sName = {"ane_SD_EMG_1054"}

conditions = {"awake_rest_ec", "awake_rest_eo", "SED_1_rest", "SED_2_rest", "SED_3_rest"}
# conditions = {"_awake_rest_ec", "_SED_3_rest"}
# conditions = {"_SED_3_rest"}
ending = "_clean_epo.fif"

result = {
    'SubID':sName,
    'N_Epochs': np.zeros(len(sName)),
    'N_Comps': np.zeros(len(sName)),
    'SED_1_rest': np.zeros(len(sName)),
    'SED_2_rest': np.zeros(len(sName)),
    'SED_3_rest': np.zeros(len(sName))}
 
NepochsDF = pd.DataFrame(result, index = sName, columns=['awake_rest_ec','awake_rest_eo', 'SED_1_rest', 'SED_2_rest', 'SED_3_rest'])
NcompsDF = pd.DataFrame(result, index = sName, columns=['awake_rest_ec','awake_rest_eo', 'SED_1_rest', 'SED_2_rest', 'SED_3_rest'])
NbadchansDF = pd.DataFrame(result, index = sName, columns=['awake_rest_ec','awake_rest_eo', 'SED_1_rest', 'SED_2_rest', 'SED_3_rest'])


for sub in sName:
    for cond in conditions:
        # 0.1 Decide which participant and which condition
        filename = sub + '_' + cond
        fullfilename = sub + '_' + cond + ending
        #filepath = Path("C:/Users/imadjb/Documents/EEG_ANALYSIS/ane_SD_1016")
        # outpath = "E:/Anesthesia/EEG_preProcessed/" + sub + "/"
        filepath = ("E:/Anesthesia/EEG_preProcessedAuto/" + sub + "/" + fullfilename)

        
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
        NepochsDF.loc[[sub],[cond]] = data.info['bad_epochs']
        NcompsDF.loc[[sub],[cond]] = data.info['n_rejected_comps']
        NbadchansDF.loc[[sub],[cond]] = data.info['n_bad_chan']
