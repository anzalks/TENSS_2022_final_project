from open_ephys.analysis import Session
import matplotlib.pylab as plt
file_path ='/Users/anzalks/Documents/tenss/Final_project/ep_data_analysis/ep_data/2022-06-17_16-11-08/'
session = Session(file_path)
recordnode = session.recordnodes[0]
print(session)
