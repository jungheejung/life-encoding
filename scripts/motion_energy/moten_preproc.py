# %%
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
from scipy import signal
from scipy.stats import pearsonr
# %%
moten_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/motion_energy' 
moten_dir = '/Volumes/life-encoding/data/motion_energy' 
data_dir = '/dartfs/rc/lab/D/DBIC/DBIC/life_data/'
data_dir = '/Volumes/life_data/'
model_dir = os.path.join(data_dir, 'w2v_feature')

# %%
resample = []
for run in [1,2,3,4]:
    print(f"----------------run: {run}--------------------")
    moten = np.load(join(moten_dir, f'moten_run-{run}.npy'))


    # step 1: log transform 
    print(f"* values smaller than 0: {np.sum(moten < 0)}, \n* moten size: {moten.size}, \n* proportion of negative moten values: {np.sum(moten<0)/moten.size}")
    # given that we have few negative numbers, we'll just convert them to 0 
    moten_log = np.nan_to_num(np.log(moten))
    print(f"* mean:{np.mean(moten_log)}, \n* max: {np.max(moten_log)}, \n* min: {np.min(moten_log)}, \n* std: {np.std(moten_log)}")
    print(f"* mean axis 0: {np.mean(moten_log, axis=0)}, \n* std axis 0: {np.std(moten_log, axis = 0)}")
    # step 2: down sample
    # method 1, scipy _________________________________________________________________
    fps = 30
    TR = 2.5
    fmri_rate = 1/TR

    number_of_samples = np.ceil(moten_log.shape[0] * fmri_rate / fps)
    resample_scipy = signal.resample(moten_log, int(number_of_samples), axis = 0)
    print(f"scipyresample shape: {resample_scipy.shape}")
    np.save(os.path.join(moten_dir, f'resample-scipy_run-{run}.npy'), resample_scipy)
    # scipy plots
    # fig, axes = plt.subplots(ncols=1, figsize=(30, 10));    plt.plot(resample_scipy)
    # plt.title("downsampled via scipy.signal")
    # fig.savefig(os.path.join(moten_dir, f'resample-scipy_run-{run}.png'),dpi = 300)

    # method 2: average over time window (TR) ___________________________________________
    # nframes = moten_log.shape[0]
    # nTRs = np.round(nframes/fps/TR)
    # downsamp = []
    # for i in np.arange(nTRs)*fps*TR:
    #     downsamp.append(np.mean(moten_log[int(i):int(i+TR*fps)], axis=0))
    # resample_avg = np.vstack(downsamp)
    # np.save(os.path.join(moten_dir, f'resample-averageTR_run-{run}.npy'), resample_avg)
    # print(f"average resample shape: {np.array(resample_avg).shape}")

    # fig, axes = plt.subplots(ncols = 1, figsize = (30, 10));    plt.plot(np.array(resample_avg))
    # plt.title("downsampled via averaging")
    # fig.savefig(os.path.join(moten_dir, f'resample-averageTR_run-{run}.png'), dpi = 300)

    # r = []
    # for i in np.arange(resample_scipy.shape[1]):
    #     r.append(pearsonr(resample_scipy[:,i], resample_avg[:,i])[0])
    # print(f"run {run} correlation: {r}")

    # vstack _____________________________________________________________________________
    resample.append(resample_scipy)

    


# %%
resample_all = np.vstack(resample)
model_durs = {1: 369, 2: 341, 3: 372, 4: 406}
# assert np.sum(model_durs.values()) == resample_all.shape[0]

# NOTE: The last run (run-4) has one more TR. 
# We're explicitly removing a TR, due to rounding
resample_remedy = resample_all[:-1]
assert np.sum(list(model_durs.values())) == resample_remedy.shape[0]

np.save(os.path.join(model_dir, 'visual_moten.npy'), resample_all)
# %%
