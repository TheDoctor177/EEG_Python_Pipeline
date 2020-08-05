# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:59:05 2020

@author: ReneS
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:13:41 2020

@author: ReneS
"""

import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
from pci_st import *
from scipy.integrate import simps
from autoreject import AutoReject


# define path and filename (here you might want to loop over datasets!)
sName = {"ane_SD_EMG_1010", "ane_SD_EMG_1016", "ane_SD_EMG_1017", "ane_SD_EMG_1022", "ane_SD_EMG_1024", "ane_SD_EMG_1033", "ane_SD_EMG_1036", "ane_SD_EMG_1045", "ane_SD_EMG_1046", "ane_SD_EMG_1054"}
# sName = {"ane_SD_EMG_1054"}

conditions = {"awake_tms", "SED_1_TMS", "SED_2_TMS", "SED_3_TMS"}
# conditions = {"_awake_rest_ec", "_SED_3_rest"}
# conditions = {"_SED_3_rest"}
ending = "_epo.fif"

# Make datafram
result = {
    'SubID':sName,
    'awake_TMS': np.zeros(len(sName)),
    'SED_1_TMS': np.zeros(len(sName)),
    'SED_2_TMS': np.zeros(len(sName)),
    'SED_3_TMS': np.zeros(len(sName))}
 
PCI_st = pd.DataFrame(result, index = sName, columns=['awake_tms', 'SED_1_TMS', 'SED_2_TMS', 'SED_3_TMS'])

# Plot butterfly?
plot = False

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
            data.apply_baseline()
        except:
            print("**********************************************************")
            print("File %s does not exist" % filepath)
            print("**********************************************************")
            continue
        
        n_epochs_before = data.get_data().shape[0]
        
        ar = AutoReject()
        epochs_clean = ar.fit_transform(data)
        
        epochs_clean.info['bad_epochs'] = n_epochs_before - data.get_data().shape[0]
        
        # Save clean epochs here
        
        # 12. create evoked data (average over all trials)/ Butterfly plot
        evoked_epochs = epochs_clean.average(picks='eeg')
        if plot:
            evoked_epochs.plot_joint(title=filename) # plots butterfly plot
            while not plt.waitforbuttonpress():            
                print('Inspecting Butterfly..')
            plt.close('all')
            
        # 13. Calculate PCI_ST
        # Might be nice to directly put into a table...
        # Use same baseline window as above, define response window from evoked response!
        par = {'baseline_window':(-400,-0.007), 'response_window':(0.007,300), 'k':1.2, 'min_snr':1.1, 'max_var':99, 'embed':False,'n_steps':100}
        pci = calc_PCIst(evoked_epochs.data, evoked_epochs.times, **par)
        
        PCI_st.loc[[sub],[cond]] = pci