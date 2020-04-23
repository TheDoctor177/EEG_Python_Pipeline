import mne
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import simps
import os
# os.chdir("//lagringshotell/imb-jstormlab/Data/Anesthesia_Project/EEG_Analysis")
# from pci_st import *

# !!!!! Change paths in line 112/113 to desired paths !!!!!


# Define some functions
        
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
    if type(eeg) is mne.epochs.EpochsArray:
        # Benny's way is way too slow.... and a bit ugly...         
        # Let's try it MNE style
        n_chan = eeg.ch_names.__len__()
        n_disp = 8
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


# define path and filename (here you might want to loop over datasets!)
# sName = {"ane_SD_EMG_1010", "ane_SD_EMG_1016", "ane_SD_EMG_1017", "ane_SD_EMG_1022", "ane_SD_EMG_1024", "ane_SD_EMG_1033", "ane_SD_EMG_1036"}
sName = "ane_SD_EMG_1045"
if sName == "ane_SD_EMG_1016":
    conditions = {"_awake_tms", "_sed_tms_1", "_sed_tms_2", "_sed_tms_3"}
else:
    conditions = {"_awake_tms", "_SED_1_TMS", "_SED_2_TMS", "_SED_3_TMS"}
ending = ".vhdr"

# Loop through desired subjects and conditions
# for sub in sName:
for cond in conditions:
    # 0.1 Decide which participant and which condition
    FileNoEnding = sName + cond
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
    
    # 2. Replace TMS pulse artefact with noise based on baseline statistics
    events = mne.events_from_annotations(data) #Generate an event file for pulses and annotations:
    mne.preprocessing.fix_stim_artifact(data, events=events[0], event_id=events[1]['Response/R128'], tmin=-0.002, tmax=0.007, mode='linear')
    
    # 3. channel info (remove EMG and set type for EOG channels)
    #data.drop_channels('EMG')
    data.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
    data.set_montage('standard_1005')
    
    # 4. resample (with low-pass filter!)
    new_sampling = 1000
    data.resample(new_sampling, npad='auto')
    #plot_response(data, ['time', 'psd'])
    
    # 5. filter (first high- then low-pass; notch-filter?)
    l_cut, h_cut = 0.5, 45
    data.filter(l_freq=l_cut, h_freq=None)
    #data.notch_filter(freqs=50)
    # plot_response(data, 'psd')
    
    
    # 6. remove bad channels (or do not remove but track them)
    good, bad = detect_bad_ch(data)
    data.info['bads'] = bad  # keep track of bad channels but do not remove (MNE style)
    #data.drop_channels(bad)  # remove bad channels (eeglab style)
    data = data.interpolate_bads(reset_bads=True)  # for presentation of bad channels change to False
    
    # Low pass filter after bad chan detect cause easier to see bad chan
    data.filter(l_freq=None, h_freq=h_cut, h_trans_bandwidth=4)
    # data.notch_filter(freqs=[50], method='spectrum_fit', p_value=1)
    # data.notch_filter(freqs=np.arange(50, h_cut, 50))
    
    # 7. Epoching data (important before ICA)
    #Epoch data using function (setting index 0 for events selects a list in the array obj. events):
    events = mne.events_from_annotations(data) #Generate an event file for pulses and annotations:
    data = mne.Epochs(data, events[0], event_id=events[1]['Response/R128'], tmin=-1, tmax=2, preload=True, baseline=None) # No baseline applied
    
        
    # 8. PCA + ICA (by default if rank violated)
    n_ic = len(data.ch_names)-len(bad)
    ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True), max_pca_components=n_ic)
    ica.fit(data, picks=['eeg', 'eog'])
    
    ica.plot_components(inst=data)  # show all components interactive (slow)
    # Wait until any key is pressed in the last opened window!!!!
    # !!!! DO NOT CLOSE WINDOWS MANUALLY!!!
    while not plt.waitforbuttonpress():            
        print('Inspecting channels..')
    plt.close('all')
    
    # 9. loop through each channel (faster):
    # ica.exclude = detect_bad_ic(ica, data)
    clean_data = data.copy()
    ica.apply(clean_data, exclude=ica.exclude)
    
    # 10. Epoch into shorter epochs and apply baseline correction (in accordance with paper) 
    clean_data.crop(tmin=-0.4, tmax=0.4)
    clean_data.apply_baseline(None,0)
    
    # # 11. Lowpass filter again
    # h_cut = 45
    # data.filter(l_freq=None, h_freq=h_cut)
    
    # 10. Run ICA again to remove any remaining artifacts?
    
    # 11. re-reference to average
    clean_data.set_eeg_reference('average', projection=False)  # you might want to go with True
    
    clean_data.save((outpath + FileNoEnding + '_epo.fif'))

    
    # Inspect epochs and drop bad ones
    clean_data.plot(n_epochs=5, n_channels=16)
    while not plt.waitforbuttonpress():            
        print('Inspecting channels..')
    plt.close('all')
    clean_data.drop_bad()
    
    # 11.1 Save epoched data
    clean_data.save((outpath + FileNoEnding + '_clean_epo.fif'))
    
    # 12. create evoked data (average over all trials)/ Butterfly plot
    # evoked_epochs = clean_data.average()
    # evoked_epochs.plot_joint() # plots butterfly plot

    # 13. Calculate PCI_ST
    # Might be nice to directly put into a table...
    # Use same baseline window as above, define response window from evoked response!
    # par = {'baseline_window':(-400,-0.002), 'response_window':(0.007,50), 'k':1.2, 'min_snr':1.1, 'max_var':99, 'embed':False,'n_steps':100}
    # pci = calc_PCIst(evoked_epochs.data, evoked_epochs.times, **par)
    # print("**********************************************************")
    # print("**********************************************************")
    # print(pci)
    # print("**********************************************************")
    # print("**********************************************************")
