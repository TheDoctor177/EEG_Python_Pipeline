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
            ax1.set_xlim([0, 80])
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
# sName = {"ane_SD_EMG_1010", "ane_SD_EMG_1016", "ane_SD_EMG_1017", "ane_SD_EMG_1022", "ane_SD_EMG_1024", "ane_SD_EMG_1033", "ane_SD_EMG_1036"}
sName = "ane_SD_EMG_1054"
if sName == "ane_SD_EMG_1010":
    conditions = {"_awake_rest_ec", "_awake_rest_eo", "_SED_1", "_SED_2", "_SED_3"}
elif sName == "ane_SD_EMG_1016":
    conditions = {"_awake_rest_ec", "_awake_rest_eo", "_sed_1", "_sed_2", "_sed_3"}
else:
    conditions = {"_awake_rest_ec", "_awake_rest_eo", "_SED_1_rest", "_SED_2_rest", "_SED_3_rest"}
    # conditions = {"_awake_rest_ec", "_SED_3_rest"}
    # conditions = {"_SED_3_rest"}
ending = ".vhdr"
# Settings
new_sampling = 1000
l_cut, h_cut = 1, 45

# for sub in sName:
for cond in conditions:
    # 0.1 Decide which participant and which condition
    filename = sName + cond
    fullfilename = sName + cond + ending
    #filepath = Path("C:/Users/imadjb/Documents/EEG_ANALYSIS/ane_SD_1016")
    outpath = "E:/Anesthesia/EEG_preProcessed/" + sName + "/"
    filepath = Path(("E:/Anesthesia/EEG/" + sName))
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
    if data.info['ch_names'][-1] == 'EMG':
        data.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog', 'EMG': 'emg'})
    else:
        data.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
    data.set_montage('standard_1005')
    # Drop EMG channel if present (might have forgotten to change workspace during recording)
    if data.info['ch_names'][-1] == 'EMG':
        data.drop_channels(data.info['ch_names'][-1])
    
    # 2. resample (with low-pass filter!)
    data.resample(new_sampling, npad='auto')
    #plot_response(data, ['time', 'psd'])
    
    # High-pass filter
    data.filter(l_freq=l_cut, h_freq=None, picks=['eeg', 'eog'])
    
    good, bad = detect_bad_ch(data)
    data.info['bads'] = bad  # keep track of bad channels but do not remove (MNE style)
    data.bad_chan = bad # Keep track of bad channels in main data
    data.n_bad_chan = len(bad) # Keep track of number of channels
    #data.drop_channels(bad)  # remove bad channels (eeglab style)
    data.interpolate_bads(reset_bads=True) 
    
    # 3. Filter for ICA; also notch cause it fucks up ICA....
    data.filter(l_freq=None, h_freq=45, picks=['eeg', 'eog'])
    data.notch_filter(freqs=(np.arange(50,80,50)), picks=['eeg', 'eog'])
    # remove line-noise by sinusoidal fit (takes longer but worth it!)
    # data.notch_filter(freqs=[50], method='spectrum_fit', p_value=1)
    # plot_response(data, 'psd')
    
    # 4. remove bad channels (or do not remove but track them)
    #good, bad = detect_bad_ch(data)
    # eeg_data = data.copy().pick_types(eeg=True, eog=False, emg=False)
    # good, bad = detect_bad_ch(data)
    # data.info['bads'] = bad  # keep track of bad channels but do not remove (MNE style)
    # data.bad_chan = bad # Keep track of bad channels in main data
    # data.n_bad_chan = len(bad) # Keep track of number of channels
    # #data.drop_channels(bad)  # remove bad channels (eeglab style)
    # data.interpolate_bads(reset_bads=True)  
    
    # Append interpolated data to data
    # data.drop_channels(data.ch_names[0:62])

    
    # 5. PCA + ICA (by default if rank violated)
    n_ic = len(data.ch_names)-len(bad)
    if not bad:
        ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True))
    else:
        ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True), max_pca_components=n_ic)
    ica.fit(data, picks=['eeg', 'eog'])
    
    ica.plot_components(inst=data)  # show all components interactive (slow)         
    while not plt.waitforbuttonpress():            
        print('Inspecting components..')
    plt.close('all')
    
    # Keep track of rejected components
    data.info['n_rejected_comps'] = len(ica.exclude)

    # Clean data from bad ICs
    clean_data = data.copy()
    ica.apply(clean_data, exclude=ica.exclude)     
    
    # 7. filter again at 45
    # clean_data.filter(l_freq=None, h_freq=h_cut)
    # plot_response(data, 'psd')
    
    # 8. re-reference to average
    clean_data.set_eeg_reference('average', projection=False)  # you might want to go with True
    
    # 9 Epoch for complexity measures
    eData = epoch(clean_data)
    eData.save((outpath + filename + '_epo.fif'), overwrite=True)

    # To keep track of bad epochs
    n_epochs_before = data.get_data().shape[0]

    # eData.drop_bad()
    # Detect and reject bad epochs manually
    # eData.plot(n_epochs=5, n_channels=16)
    # while not plt.waitforbuttonpress():            
    #     print('Inspecting channels..')
    # plt.close('all')
    # eData.drop_bad()
    
    # reject automatically
    ar = AutoReject()
    epochs_clean = ar.fit_transform(data)
    
    # number of bad epochs
    epochs_clean.info['bad_epochs'] = n_epochs_before - data.get_data().shape[0]

    # Save epoched data
    eData.save((outpath + filename + '_clean_epo.fif'), overwrite=True)

# # 15. Calculate LZC
# finData = eData.get_data(picks = 'eeg')
# resultLZ = pc.LZc(finData)

# # The following need 3D arrays to function...
# resultSCE = pc.SCE(finData)
# resultACE = pc.ACE(finData)



# remove line-noise by notch filter (not always recommended!)
#data.notch_filter(freqs=np.arange(50, h_cut, 50))
