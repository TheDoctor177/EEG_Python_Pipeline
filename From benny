import mne
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import simps


# define path and filename (here you might want to loop over datasets!)
filename = "sleep_sd1020_eyes_closed2_060219.vhdr"
filepath = Path("/home/benjamin/Downloads/sd1020/")
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
        ax1.psd(time_data, new_sampling, new_sampling)
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

    for c in range(ica_data.get_components().shape[0]):
        """loop over each channel and plot to decide if bad"""
        ica_data.plot_properties(inst=data_orig, picks=c)

        if not plt.waitforbuttonpress():
            good_ic.append(c)
            plt.close()
        else:
            bad_ic.append(c)
            plt.close()

    [bad_list.append(ica_data.ch_names.index(ci)) for ci in bad]
    return bad_list


# 1. load data (vhdr file)
data = mne.io.read_raw_brainvision(file)
data.load_data()
# plot_response(data, 'time')


# 2. TMS-artifact interpolation!!!! (here you want to remove the TMS artifact and interpolate the missing samples)


# 3. resample (with low-pass filter!)
new_sampling = 1000
data.resample(new_sampling, npad='auto')
# plot_response(data, ['time', 'psd'])


# 4. filter (first high- then low-pass; notch-filter?)
l_cut, h_cut = 1, 80
data.filter(l_freq=l_cut, h_freq=h_cut)
# plot_response(data, 'psd')

# remove line-noise by notch filter (not always recommended!)
data.notch_filter(freqs=np.arange(50, h_cut, 50))

# remove line-noise by sinusoidal fit (takes longer but worth it!)
# data.notch_filter(freqs=[50], method='spectrum_fit', p_value=1)


# 5. channel info (remove EMG and set type for EOG channels)
data.drop_channels('EMG1')
data.set_channel_types({'VEOGl': 'eog', 'VEOGu': 'eog'})
data.set_montage('standard_1005')


# 6. remove bad channels (or do not remove but track them)
good, bad = detect_bad_ch(data)
data.info['bads'] = bad  # keep track of bad channels but do not remove (MNE style)
# data.drop_channels(bad)  # remove bad channels (eeglab style)


# 7. interpolate channels (spherical spline method recommended)
data = data.interpolate_bads(reset_bads=True)  # for presentation of bad channels change to False
# plot_response(data, 'butter')


# 8. re-reference to average
data.set_eeg_reference('average', projection=False)  # you might want to go with True


# 9. PCA + ICA (by default if rank violated)
n_ic = data._data.shape[0]-len(bad)
ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True), max_pca_components=n_ic)
ica.fit(data, picks=['eeg', 'eog'])

# ica.plot_components(inst=data_reref)  # show all components interactive (slow)

# loop through each channel (faster):
ica.exclude = detect_bad_ic(ica, data)
clean_data = data.copy()
ica.apply(clean_data)


# 10. Compute power
#
# 1st: Area under the curve in Welch transform)
# f_power = [8, 12]
# win = 2 * clean_data.info['sfreq'].__int__()  # 2s window
# freqs, psd = scipy.signal.welch(ica._data, ica.info['sfreq'].__int__(), nperseg=win)
# freq_res = freqs[1] - freqs[0]
# idx_power = np.logical_and(freqs >= f_power[0], freqs <= f_power[1])
#
# # absolute power for each channel
# power = []
# [power.append(simps(psd[i][idx_power], dx=freq_res)) for i in range(psd.shape[0])]


# 2nd: bandpass filter and squaring (be carefull with epochs)

# 3rd: Complex Morlet Wavelet transform


"""create sine wave for fun"""
# import numpy as np
# import matplotlib.pyplot as plt
#
# A = 2
# f = 10
# s = 100
# t = np.arange(0, 1, 1/s)
# ph = 0.25
#
# x = A*np.sin(2*np.pi*f*t+ph)
# plt.plot(t, x)
# plt.show()
