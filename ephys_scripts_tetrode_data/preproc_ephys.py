from open_ephys.analysis import Session
import ephys_functions as epf
import matplotlib.pyplot as plt
import scipy.signal as spy
import numpy as np
import pandas as pd
import pdb

# load data
directory = '/Users/anzalks/Desktop/last_minute_data/ep_data/2022-06-17_16-11-08' # string path to the folder with recordings
sess      = Session(directory)
data      = np.array(sess.recordnodes[0].recordings[0].continuous[0].samples) # access recordings
b_data    = pd.read_csv('/Users/anzalks/Desktop/last_minute_data/mouse2022-06-17T16_12_52_with_header.csv') # reads csv from behaviour data
fs        = 120 # frame rate of the camera

# Filter and downsample the data
''' 
17 and 25 are the ephys channels in this particular dataset. This segment downsamples and extracts traces from channels described. After that different filters are applied. Based on use case you can choose your filter properties
'''
n_data = []
for chan_id in range(17, 25):
    chan_data = data[:, chan_id]
    chan_data = epf.filter_data(chan_data, 300, 'lowpass', 30000)
    chan_data = epf.downsampling_funct(chan_data, 30000, 1000)
    chan_data = epf.filter_data(chan_data, 0.1, 'highpass', 1000)
    chan_data = epf.filter_data(chan_data, [49.5,50.5], 'bandstop',1000)
    chan_data = epf.filter_data(chan_data, [99.5,100.5], 'bandstop',1000)
    chan_data = epf.filter_data(chan_data, [149.5,150.5], 'bandstop',1000)
    n_data.append(chan_data)

filt_file = np.transpose(np.array(n_data))

"""
The behaviour data is a time series with some variables changing over time and in this case it's a door opening and closing. By taking the differential we are able to find the points which are showing maximum change and thus the events/time points where the door opened/closed
"""
# compute derivative of x and y by shifting by 1 element and subtracting
x = epf.smooth(np.array(b_data['Door_X']), 20)
x = x[:-1] - x[1:]
y = epf.smooth(np.array(b_data['Door_Y']), 20)
y = y[:-1] - y[1:]
d = (np.sqrt(np.square(x) + np.square(y))>5).astype(np.int32)

# clean up d
for i in range(len(d) - 1):
    if d[i] == 1:
        if d[i + 1] == 0:
            d[i]=0
d[37379] = 1
d[37380] = 1

# extract opening times from derivative
d = d[:-1] - d[1:]
ind = np.squeeze(np.argwhere(d==-1)) # -1 corresponds to event onset (in this case DOOR OPEN onset)
ind = ind/fs

# separate opening and closing times
ind_o = ind[0:ind.shape[0]:2]
ind_c = ind[1:ind.shape[0]:2]

# generate event codes
tr_type = np.ones(ind_o.shape[0] + ind_c.shape[0])
tr_type[:ind_o.shape[0]] = 2
ev_codes = np.ones(ind_o.shape[0] + ind_c.shape[0])

# recreate event times
ev_times = np.hstack([ind_o, ind_c])
ind = np.argsort(ev_times)
ev_times = ev_times[ind][:-1]
ev_codes = ev_codes[ind][:-1]
tr_type  = tr_type[ind][:-1]

# find time of first camera frame
ind = np.argwhere(data[:,chan_id]<10000)[0]/30000
ev_times = ev_times + ind
ev_times = np.floor(ev_times * 1000).astype(np.int32) #convert time to samples

# extract trials from channel
trials = epf.data_to_event(filt_file, ev_codes, ev_times, 1, 1000, 10000)
np.save('/Users/anzalks/Documents/tenss/Final_project/ephys_scripts_from_andrei/temp_data/trial_data.npy', trials)
np.save('/Users/anzalks/Documents/tenss/Final_project/ephys_scripts_from_andrei/temp_data/ev_type.npy',tr_type)
