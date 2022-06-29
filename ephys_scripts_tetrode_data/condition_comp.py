import os
import numpy as np
import scipy.signal as spy
import scipy.fft as fft
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
import pdb
import pandas as pd

"""
we have started with the frequency 0.1 one so we use 0.1 as starting
frequency here and go upto maximum frequency in the range of our fourier
analysis with a step size of our frequency bins
"""
def bin_to_sampling(bin_size,no_samples,sampling_freq):
    max_freq=int(np.floor(sampling_freq/2))
    no_freq_bin=int(np.floor(no_samples/2)+1)
    freq_range = np.linspace(0.1,max_freq,no_freq_bin)
    return freq_range
"""
read csv file with pandas
"""
#trial_info = pd.read_csv('/Users/anzalks/Documents/tenss/Final_project/probe_recording/datasets/M086_GENUS_Mk2_Static-7-60_V_5/trial_info.csv')

"""
load spectral data as np array
"""
spectral_data =np.load('/Users/anzalks/Documents/tenss/Final_project/ephys_scripts_tetrode_data/temp_data/spectral_data.npy')

spectral_data = np.log10(spectral_data[0, :, :, :])
#spectral_data = spectral_data[0,:,:,:]

"""
the window size used for computing the spetra s 256 and it's moved over 10 sample gradation. hence we took 256 and 10 for calculating number of windows(n_win)
for number of smaples in prevent of 1000s(bl_samp) = n_win*1000/7000
"""
n_win = int(np.floor((11000-256)/10)+1)
bl_samp = int(np.floor(n_win/11))
""""
compute baseline statictics per frequency
Since we are binning our 7000 samples into 53 bins, we can use the fisrt 7 bins as our basline of 1000 samples.
"""
bl_spect = np.zeros([spectral_data.shape[1], bl_samp * spectral_data.shape[2]])
for trial in range(spectral_data.shape[2]):
    bl_spect[:, trial * bl_samp : (trial+1) * bl_samp] = np.transpose(spectral_data[:bl_samp,:, trial])
mean_bl = np.mean(bl_spect, axis=1)
std_bl = np.std(bl_spect,axis=1)
#pdb.set_trace()
for freq in range(spectral_data.shape[1]):
    spectral_data[:, freq, :] = (spectral_data[:, freq, :] - mean_bl[freq]) / std_bl[freq]
"""
Loads the numpy array saved from events. you can also give any time series with a tag manually of event and time point in this place as well.
"""
flicker = np.load('/Users/anzalks/Documents/tenss/Final_project/ephys_scripts_tetrode_data/temp_data/ev_type.npy')#np.array(trial_info['FlickerFrequency'])
#identify only unique events in the event array
unq = np.unique(flicker)
#our dounsampled sampling rate is 1000hz and bin size is 2Hz, number of bins is 256. This we obtained from the pectral data generation script
freq_range = bin_to_sampling(2,256,1000)
#maping of tick values for plotting to the actual values we plot on the graph
y_tick_loc =np.array(range(len(freq_range)))[0:len(freq_range):10][:6]
y_tick_val = np.round(freq_range[0:len(freq_range):10][:6]).astype(int)
x_tick_loc =np.array(range(0,1000,100))
x_tick_val =(x_tick_loc/100)-1

#Since all frequencies are not relevant in this case we plot only upto 128Hz hence picked only 64 bins
n_row, n_col = (1,2)
f, ax = plt.subplots(n_row, n_col, figsize=(10,7))
for idx, c_val in enumerate(unq):
    trial_idx = np.squeeze(np.argwhere(flicker==c_val))
    c_spect = spectral_data[:,:,trial_idx]
    av_spect = np.mean(c_spect,axis =2)
    av_spect = av_spect[:,:64]
    print(av_spect.shape)
    #av_spect = np.log10(av_spect)
    #plt.figure()
    idx = np.squeeze(np.unravel_index(idx, [n_row, n_col]))
    if n_row ==1:
        idx = idx[1]
    #pdb.set_trace()
    ax[idx].imshow(np.transpose(av_spect),origin='lower', aspect='auto',
               cmap='jet',interpolation = 'bicubic')
    ax[idx].axvline(bl_samp-3,color='w',lw=2)
    ax[idx].axvline(bl_samp+3,color='w',lw=2)
    ax[idx].axvline(bl_samp,color='k',lw=2)
    ax[idx].set_xlabel('Time (s)')
    ax[idx].set_ylabel('Frequency Hz', fontsize=10)
    ax[idx].set_title(f'Average power spectra of {c_val} Hz visual stimulation',fontsize=10)
    ax[idx].set_yticklabels(y_tick_val)
    ax[idx].set_yticks(y_tick_loc)
    ax[idx].set_xticklabels(x_tick_val)
    ax[idx].set_xticks(x_tick_loc)

f.suptitle(f'Power spectra of multiple freq stimulations', fontsize=16)
#f.tight_layout()
#f.delaxes(ax[3,1])
plt.subplots_adjust(left =0.07, bottom=0.05, right=0.97, top=0.92, wspace=0.125, hspace=0.5)
plt.show()
"""
f, ax = plt.subplots(1,2)
for idx, c_val in enumerate(unq):
    trial_idx = np.squeeze(np.argwhere(flicker==c_val))
    c_spect = spectral_data[:,:,trial_idx]
    av_spect = np.mean(c_spect,axis =2)
    #av_spect = np.log10(av_spect)
    #plt.figure()
    #idx = np.unravel_index(idx, [4, 2])
    ax[idx].imshow(np.transpose(av_spect),origin='lower', aspect='auto',
               cmap='jet',interpolation = 'bicubic')
    ax[idx].axvline(bl_samp-1,color='w',lw=2)
    ax[idx].axvline(bl_samp+1,color='w',lw=2)
    ax[idx].axvline(bl_samp,color='k',lw=2)
    ax[idx].set_xlabel('bins (512 samples in one bin)')
    ax[idx].set_ylabel('frequency bins (257/Nq)', fontsize=8)
    ax[idx].set_title(f'Average power spectra of {c_val} Hz visual stimulation',fontsize=10)




f.suptitle(f'Power spectra of multiple freq stimulations', fontsize=16)
#f.tight_layout()
plt.show()
#pdb.set_trace()
"""
