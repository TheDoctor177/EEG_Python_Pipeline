import mne
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import simps


# define path and filename (here you might want to loop over datasets!)
filename = "ane_SD_EMG_1016_awake_tms.vhdr"
#filepath = Path("C:/Users/imadjb/Documents/EEG_ANALYSIS/ane_SD_1016")
filepath = Path("E:/Anesthesia/EEG/ane_SD_1016")
file = filepath / filename

        
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


    
# Make feature that asks for parameters by inspecting the data
def TMS_interpolate(data,tmin=-0.2,tmax=0.5,fsample = 5000, pulse_start = -0.002, pulse_end = 0.007): # interpolates TMS pulses
    events = mne.events_from_annotations(data) #Generate an event file for pulses and annotations:

    #Epoch data using function (setting index 0 for events selects a list in the array obj. events):
    "Baseline correct??"
    baseline = (None, 0.0)
    epochs = mne.Epochs(data, events[0], tmin=tmin, tmax=tmax, preload=True)
    #epochs.plot()

    pulse_datapoints = (pulse_end*fsample)+(np.absolute(pulse_start*fsample)) #number of datapoints included in pulse
    pulse_start_datapoint = (np.abs(tmin)*fsample)-(np.abs(pulse_start)*fsample) # datapoint at which pulse begins
    pulse_end_datapoint = (np.abs(pulse_end)*fsample)+(np.abs(tmin)*fsample) #datapoint which pulse ends
  
    raw_epochs=epochs.get_data() # generates a datafile of the epochs
    raw_baseline = raw_epochs[:,:,0:int(pulse_start_datapoint)] # defines the baseline from raw EEG data from before pulse
    mean_baseline = np.mean(raw_baseline,axis=2) # Here we take the mean of the datapoints
    std_baseline = np.std(raw_baseline,axis=2) #we get std


    new_pulse = [] # we generate data to write over EEG during pulse
    for i in range(0,int(pulse_datapoints)): # is defined after number of datapoints we want to do it for. that means number of datpapoints in our pulse
        randomsample = np.random.normal(loc=mean_baseline, scale=std_baseline) #generate random datapoints with mean and std.
        new_pulse.append(list(randomsample)) #creates a list of datapoints for all 300*64 time periods during pulse

    new_pulse = np.array(new_pulse) #generate array
    new_pulse = np.transpose(new_pulse, axes=(1,2,0)) #change order of array

    raw_epochs_interpolated = raw_epochs.copy()
    raw_epochs_interpolated[:,:,int(pulse_start_datapoint):int(pulse_end_datapoint)] = new_pulse

    newepochs = mne.EpochsArray(raw_epochs_interpolated,info=epochs.info, tmin=tmin, baseline = baseline)
    return newepochs

    #plt.figure()
    #plt.plot(raw_epochs[100,10,:])
    #plt.plot(raw_epochs_interpolated[100,10,:])
    #plt.show()


# 1. load data (vhdr file)
data = mne.io.read_raw_brainvision(file)
data.load_data()
# plot_response(data, 'time')

# 1.1. channel info (remove EMG and set type for EOG channels)
#data.drop_channels('EMG')
data.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
data.set_montage('standard_1005')

# 2. Replace TMS pulse artefact with noise based on baseline statistics
data = TMS_interpolate(data) # look at function to change
# data.apply_baseline(None,0.0) ; Done in function now

# 3. remove bad channels (or do not remove but track them)
#good, bad = detect_bad_ch(data)
data = detect_bad_ch(data)
bad = data.info['bads']  # keep track of bad channels but do not remove (MNE style)
data.bad_chan = bad # Save bad channels to to keep track
#data.drop_channels(bad)  # remove bad channels (eeglab style)
data = data.interpolate_bads(reset_bads=True)  # for presentation of bad channels change to False

# 4. resample (with low-pass filter!)
new_sampling = 1000
data.resample(new_sampling, npad='auto')
#plot_response(data, ['time', 'psd'])


# 7. PCA + ICA (by default if rank violated)
n_ic = len(data.ch_names)-len(bad)
ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True), max_pca_components=n_ic)
ica.fit(data, picks=['eeg', 'eog'])

ica.plot_components(inst=data)  # show all components interactive (slow)
# Wait until any key is pressed in the last opened window!!!!
# !!!! DO NOT CLOSE WINDOWS MANUALLY !!!
while not plt.waitforbuttonpress():            
    print('Inspecting channels..')
plt.close('all')

# 8. loop through each channel (faster):
# ica.exclude = detect_bad_ic(ica, data)
clean_data = data.copy()
ica.apply(clean_data, exclude=ica.exclude)
# 9. Replace TMS pulse artefact with noise based on baseline statistics (AGAIN)

# 10. filter (first high- then low-pass; notch-filter?)
l_cut, h_cut = 1, 80
clean_data.filter(l_freq=l_cut, h_freq=h_cut)
# plot_response(data, 'psd')

# 11. Replace data around TMS pulse with constant amplitude data? (research)

# 12. Run ICA again to remove any remaining artifacts.

# 14. re-reference to average
clean_data.set_eeg_reference('average', projection=False)  # you might want to go with True


# (15). create evoked data (average over all trials)/ Butterfly plot

evoked_epochs = clean_data.average()
evoked_epochs.plot_joint() # plots butterfly plot

# remove line-noise by notch filter (not always recommended!)
#data.notch_filter(freqs=np.arange(50, h_cut, 50))
