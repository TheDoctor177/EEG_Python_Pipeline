# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:56:01 2020

@author: ReneS
"""
import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sName = {"ane_SD_EMG_1010", "ane_SD_EMG_1016", "ane_SD_EMG_1017", "ane_SD_EMG_1022", "ane_SD_EMG_1024", "ane_SD_EMG_1033", "ane_SD_EMG_1036", "ane_SD_EMG_1045", "ane_SD_EMG_1046", "ane_SD_EMG_1054"}
# sName = {"ane_SD_EMG_1054"}

conditions = {"awake_rest_ec", "awake_rest_eo", "SED_1_rest", "SED_2_rest", "SED_3_rest"}
# conditions = {"_awake_rest_ec", "_SED_3_rest"}
# conditions = {"_SED_3_rest"}
ending = "_clean_epo.fif"


for sub in sName:
    for cond in conditions:
        # 0.1 Decide which participant and which condition
        filename = sub + '_' + cond
        fullfilename = sub + '_' + cond + ending
        #filepath = Path("C:/Users/imadjb/Documents/EEG_ANALYSIS/ane_SD_1016")
        # outpath = "E:/Anesthesia/EEG_preProcessed/" + sub + "/"
        filepath = ("E:/Anesthesia/EEG_preProcessed/" + sub + "/" + fullfilename)
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
        except:
            print("**********************************************************")
            print("File %s does not exist" % filepath)
            print("**********************************************************")
            continue
        
        data.plot_psd(fmax=80)
        while not plt.waitforbuttonpress():            
            print('Inspecting channels..')
        plt.close('all')