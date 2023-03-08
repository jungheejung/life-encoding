import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
from scipy import signal
moten_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/motion_energy' 
#for run in [1,2,3,4]:
run = 1
moten = np.load(join(moten_dir, f'moten_run-{run}.npy'))
#print mean, max, min, sd


# step 1: log transform 
print(np.sum(moten < 0), moten.size, np.sum(moten<0) /moten.size)
# given that we have few negative numbers, we'll just convert them to 0 

moten = np.nan_to_num(np.log(moten))

print(f"mean:{np.mean(moten)}, {np.max(moten)}, {np.min(moten)}, {np.std(moten)}")
print(f"{np.mean(moten, axis=0)}, {np.std(moten, axis = 0)}")

# step 2: down sample

# method 1, scipy
fps = 30
fmri_rate = 1/2.5

number_of_samples = round(moten.shape[0] * fmri_rate / fps)
resampled_data = signal.resample(moten, number_of_samples, axis = 0)
print(resampled_data.shape)
fig, axes = plt.subplots(ncols=1, figsize=(30, 10))
plt.plot(resampled_data)
fig.savefig(os.path.join(moten_dir, 'resample-scipyresample.png'),dpi = 300)
# method 2: average over time window (TR)
n_frames = 75
n_trs = 2.5
moten_trs = []
for onset in np.arange(0, n_frames * n_trs, n_frames):
    moten_trs.append(np.mean(moten[onset:onset + n_frames], axis=0))
print(np.array(moten_trs).shape)
fig, axes = plt.subplots(ncols = 1, figsize = (30, 10))
plt.plot(np.array(moten_trs))
fig.savefig(os.path.join(moten_dir, 'resample-averageTR.png'), dpi = 300)
