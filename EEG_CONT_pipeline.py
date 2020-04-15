# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:33:18 2020

@author: ReneS
"""

import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import pyconscious as pc
import os


        
def plot_response(signal, argument):
    """plot response to check what happened with the data"""
    if "time" in argument:
        signal.plot(duration=10, remove_dc=False)
    if "psd" in argument:
        signal.plot_psd(fmin=0, fmax=80)
    if "butter" in argument:
        signal.plot(butterfly=True, color='#00000044', bad_color='r')
    if "ica" in argument:
        signal.plot_components()


def detect_bad_ch(eeg):
    """plots each channel so user can decide whether good (mouse click) or bad (enter / space)"""
    good_ch, bad_ch = [], []
    intvl = eeg.__len__() // 20
    if type(eeg) is mne.epochs.EpochsArray or type(eeg) is mne.epochs.Epochs:
        # Benny's way is way too slow.... and a bit ugly...         
        # Let's try it MNE style
        n_chan = eeg.ch_names.__len__()
        n_disp = 4
        for i in range(0,n_chan,n_disp):
            # Choose 4 channels at a time
            cur_picks = eeg.ch_names[i:(i + n_disp)]
            fig = eeg.plot(picks=cur_picks, title='Click channel names to reject, click on epochs to reject epoch; Use mouse to move around figure, press any key to advance')

            # Wait until keyboard is pressed
            while not plt.waitforbuttonpress():            
                print('Inspecting channels..')

            plt.close(fig)
        return eeg
    else:
        for ch in eeg.ch_names:
            """loop over each channel and plot to decide if bad"""
            time_data = eeg[eeg.ch_names.index(ch)][0][0]
            df = pd.DataFrame()
            for i in range(20):
                df_window = pd.DataFrame(time_data[i * intvl:(i + 1) * intvl])
                df_window += (i + 1) * 0.0001
                df = pd.concat((df, df_window), axis=1)

            df *= 1000  # just for plotting
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(f"{ch}: mouse click for keep (good), any other key for remove (bad)")
            ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1)
            ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=3)
            ax1.psd(time_data, 5000, 5000)
            ax1.set_xlim([0, 55])
            ax2.plot(df, 'b')
            plt.show()

            if not plt.waitforbuttonpress():
                good_ch.append(ch)
                plt.close(fig)
            else:
                bad_ch.append(ch)
                plt.close(fig)

        return good_ch, bad_ch


def detect_bad_ic(ica_data, data_orig):
    """plots each independent component so user can decide whether good (mouse click) or bad (enter / space)"""
    good_ic, bad_ic = [], []
    bad_list = []
    "!!Change back to full range!!"
    for c in range((ica_data.get_components().shape[1])): 
        """loop over each channel and plot to decide if bad"""
        ica_data.plot_properties(inst=data_orig, picks=c)

        if not plt.waitforbuttonpress():
            good_ic.append(c)
            plt.close()
        else:
            bad_ic.append(c)
            plt.close()

    #[bad_list.append(ica_data.ch_names.index(ci)) for ci in bad_ic]
    return bad_ic

def epoch(data, sfreq=1000, tepoch=5, time=5, tmin=-2.5, tmax=2.5):
    # Set parameters
    time = time * 60 # time in sec
    nepochs = int((time / tepoch)) # Number of epochs
    sp = sfreq * tepoch # Samples per epoch = frequency times 5 sec.
    fiveMin =  time * sfreq # 5 minutes in sample points
    points = np.arange(0, fiveMin, sp).reshape(nepochs,1) # Indecies for event points
    dummy = np.ones((nepochs, 1), dtype=int)
    
    # Make event structure
    events = np.concatenate((points,dummy,dummy), 1)
    # Epoch data
    epoched_Data = mne.Epochs(data, events, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    
    return epoched_Data

# define path and filename (here you might want to loop over datasets!)
sName = {"ane_SD_EMG_1010", "ane_SD_EMG_1016", "ane_SD_EMG_1017", "ane_SD_EMG_1022", "ane_SD_EMG_1024", "ane_SD_EMG_1033", "ane_SD_EMG_1036"}
conditions = {"_awake_rest_ec", "_awake_rest_eo", "_SED_1_rest", "_SED_2_rest", "_SED_3_rest"}
ending = ".vhdr"
# Settings
new_sampling = 1000
l_cut, h_cut = 1, 45

for sub in sName:
    for cond in conditions:
        # 0.1 Decide which participant and which condition
        filename = sub + cond
        fullfilename = sub + cond + ending
        #filepath = Path("C:/Users/imadjb/Documents/EEG_ANALYSIS/ane_SD_1016")
        outpath = "E:/Anesthesia/EEG_preProcessed/" + sub + "/"
        filepath = Path(("E:/Anesthesia/EEG/" + sub))
        file = filepath / fullfilename
        try: 
            os.mkdir(outpath)
            print('Path created')
        except:
            print("Path exists")
        
        # 1. load data (vhdr file)
        try:
            data = mne.io.read_raw_brainvision(file)
            data.load_data()
        except:
            print("**********************************************************")
            print("File %s does not exist" % file)
            print("**********************************************************")
            continue
        # plot_response(data, 'time')
        
        # 1.1. channel info (remove EMG and set type for EOG channels)
        #data.drop_channels('EMG')
        data.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog', 'EMG': 'emg'})
        data.set_montage('standard_1005')
        
        # 3. remove bad channels (or do not remove but track them)
        #good, bad = detect_bad_ch(data)
        good, bad = detect_bad_ch(data)
        data.info['bads'] = bad  # keep track of bad channels but do not remove (MNE style)
        data.bad_chan = bad
        data.n_bad_chan = len(bad)
        #data.drop_channels(bad)  # remove bad channels (eeglab style)
        data = data.interpolate_bads(reset_bads=True)  # for presentation of bad channels change to False
        
        # 4. resample (with low-pass filter!)
        data.resample(new_sampling, npad='auto')
        #plot_response(data, ['time', 'psd'])
        
        # 5. Filter for ICA; also notch cause it fucks up ICA....
        data.filter(l_freq=1, h_freq=80)
        data.filter(freqs=50)
        # plot_response(data, 'psd')
        
        # 7. PCA + ICA (by default if rank violated)
        n_ic = len(data.ch_names)-len(bad)
        ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True), max_pca_components=n_ic)
        ica.fit(data, picks=['eeg', 'eog'])
        
        ica.plot_components(inst=data)  # show all components interactive (slow)         
        while not plt.waitforbuttonpress():            
            print('Inspecting channels..')
        plt.close('all')
        
        # 8. loop through each channel (faster):
        # ica.exclude = detect_bad_ic(ica, data)
        
        # Clean data from bad ICs
        clean_data = data.copy()
        ica.apply(clean_data, exclude=ica.exclude)     
        
        # 10. filter (first high- then low-pass; notch-filter?)
        clean_data.filter(l_freq=l_cut, h_freq=h_cut)
        # plot_response(data, 'psd')
        
        # 12. Run ICA again to remove any remaining artifacts.
        
        # 14. re-reference to average
        clean_data.set_eeg_reference('average', projection=False)  # you might want to go with True
        
        # 14.5 Epoch for complexity measures
        eData = epoch(clean_data)
        eData.drop_bad()
        # Detect and reject bad epochs
        eData.plot(n_epochs=5, n_channels=16)
        while not plt.waitforbuttonpress():            
            print('Inspecting channels..')
        plt.close('all')
        
        eData.drop_bad()
        
        # Save epoched data
        eData.save((outpath + filename + '_epo.fif'))

# # 15. Calculate LZC
# eData = eData.get_data(picks = 'eeg')
# resultLZ = pc.LZc(eData)

# # The following need 3D arrays to function...
# resultSCE = pc.SCE(eData)
# resultACE = pc.ACE(eData)



# remove line-noise by notch filter (not always recommended!)
#data.notch_filter(freqs=np.arange(50, h_cut, 50))
