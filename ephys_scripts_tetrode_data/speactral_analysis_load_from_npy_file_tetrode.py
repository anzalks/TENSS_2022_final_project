import os 
import numpy as np
import scipy.signal as spy
import scipy.fft as fft
import matplotlib.pyplot as plt 
from pathlib import Path
from pprint import pprint 
import pdb 

"""
sampling frequency = 1000Hz from down sampling

"""
fs=1000

"""
window size defines the area under consideration at given point of time(moving
window size for fourier). This should't too small, or too large, we have 1KHz 
sample Max freq= fs/2, freq_bin =( no. of samples/2)+1, 
"""
window_len = 256

"""
step size defines how much the window shifts, we don't want it to move too slow
also the steps shouldn't be larger than the window length so that we don't miss
out on the data while covering.
"""
step_size = 10

"""
a gausian function to make sure the edge effects are minimalised during fourier
multiple windows are available and this fints for our use case for now
sym- false good for spectral
sym-true good for filter desings
"""
blackman = spy.windows.blackman(window_len,sym=False)

"""
file path = file path to numpy data
event data = loads the numpy data saved from the file
""" 

file_path =Path('/Users/anzalks/Documents/tenss/Final_project/ephys_scripts_from_andrei/temp_data/trial_data.npy')
event_data = np.load(str(file_path))
event_d_before = 1000
event_data_after = 10000

"""
the function takes the data window with 1D array and multiply it with a
gaussian of preference. The signal/data array is tehn used for fourier and calculate the power of it and return the power.
"""
def fourier_dat(data,gaussian_func):
    data = data*gaussian_func
    a = fft.fft(data)
    a = np.square(np.abs(a))
    return a
"""
moving average calculation using npy convolution function
data = 1D data, 
n= moving window length
"""

def move_av(data,n):
    window = np.ones(n)/n
    resolved = np.convolve(data,window,mode='same')
    return resolved
window_idx = np.arange(0,event_data.shape[1]-window_len,step_size)
#c_trail = event_data[0,:,0]
#"""
#compute the spectrogram, define the matrix size as number of windows, number of
#frequency bins
#"""
#freq_bins =int((window_len/2) + 1)
#spect = np.zeros([window_idx.shape[0],freq_bins])
#for i in range(window_idx.shape[0]):
#    c_win = c_trail[window_idx[i]:window_idx[i]+window_len]
#    c_win = fourier_dat(c_win,blackman)
#    spect[i,:]=c_win[:freq_bins]

"""
compute the spectrogram, define the matrix size as number of windows, number of
frequency bins and for many channels we add one more dimention to spect
(ch_num) and one more for loop to fill that data.
in this case event_data has first column as channel
"""
freq_bins =int((window_len/2) + 1)
spect =np.zeros([event_data.shape[0],window_idx.shape[0],
                 freq_bins,event_data.shape[2]])
for t in range(event_data.shape[2]):
    for c in range(event_data.shape[0]):
        c_trial = event_data[c,:,t]
        for i in range(window_idx.shape[0]):
            c_win = c_trial[window_idx[i]:window_idx[i]+window_len]
            c_win = fourier_dat(c_win,blackman)
            spect[c,i,:,t]=c_win[:freq_bins]
np.save('/Users/anzalks/Documents/tenss/Final_project/ephys_scripts_from_andrei/temp_data/spectral_data.npy',spect)
av_spect = np.mean(spect,axis=0)
av_spect = np.log10(av_spect)

#"""
#do moving average in the smae window 
#"""
#freq_bins =int((window_len/2) + 1)
#spect = np.zeros([event_data.shape[0],window_idx.shape[0],freq_bins])
#for c in range(event_data.shape[0]):
#    c_trial = event_data[c,:,0]
#    for i in range(window_idx.shape[0]):
#        c_win = c_trial[window_idx[i]:window_idx[i]+window_len]
#        c_win = move_av(c_win,10) 
#        #av_spect = np.log10(av_spect)
#        spect[c,i,:]=c_win[:freq_bins]
#av_spect = np.mean(spect,axis=0)

#plt.plot(av_spect)
#
#n_win = int(np.floor((event_data.shape[1]-window_len)/step_size)+1)
#x_low = -event_d_before+(window_len/2)
#x_up = -event_d_before+(window_len/2)+step_size*n_win
#x =np.linspace(x_low,x_up,n_win)
#y= np.linspace(0,(fs/2),int((window_len/2)+1))
for p in range(3):
    plt.figure()
    plt.imshow(np.transpose(av_spect[:,:,p]),origin='lower', aspect='auto',
               cmap='jet')#,interpolation = 'bicubic')
    plt.xlabel('bins (512 samples in one bin)')
    plt.ylabel('frequency bins (257/Nq)')
    plt.title('powerspectra for one channel trial average: single event')
#plt.xticks(x)
#plt.yticks(y)
#plt.imshow(np.transpose(spect),vmin=0,vmax=1000)

##print(window_idx)
#plt.xlabel('no_samples')
#plt.ylabel('amplitude (uV)')
#plt.title('egde effect reduction by gaussian')
#plt.legend()
plt.show()
