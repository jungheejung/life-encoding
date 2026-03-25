# %%
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
from scipy import signal
from scipy.stats import pearsonr
# %%
# moten_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/motion_energy' 
# moten_dir = '/Volumes/life-encoding/data/motion_energy' 
# data_dir = '/dartfs/rc/lab/D/DBIC/DBIC/life_data/'
# data_dir = '/Volumes/life_data/'
# model_dir = os.path.join(data_dir, 'w2v_feature')

MOTEN_STEP1_DIR = '/Users/h/Documents/projects_local/life-encoding/data/revision_moten'
MOTEN_STEP2_DIR = '/Users/h/Documents/projects_local/life-encoding/data/revision_moten_resample'
from collections import OrderedDict, namedtuple
Video = namedtuple('Video', ['TR', 'fps'])
od = OrderedDict()

od['ses-01_run-01_order-02_content-wanderers']      = Video(TR=431, fps=25)
od['ses-01_run-02_order-02_content-HB']             = Video(TR=124, fps=23.98)
od['ses-01_run-03_order-01_content-huggingpets']    = Video(TR=231, fps=29.97)
od['ses-01_run-03_order-04_content-dancewithdeath'] = Video(TR=297, fps=25)
od['ses-01_run-04_order-02_content-angrygrandpa']   = Video(TR=737, fps=29.97)
od['ses-02_run-02_order-03_content-menrunning']     = Video(TR=443, fps=25)
od['ses-02_run-03_order-01_content-unefille']       = Video(TR=253, fps=25)
od['ses-02_run-03_order-04_content-war']            = Video(TR=137, fps=25)
od['ses-03_run-02_order-01_content-planetearth']    = Video(TR=322, fps=25)
od['ses-03_run-02_order-03_content-heartstop']      = Video(TR=428, fps=23.98)
od['ses-03_run-03_order-01_content-normativeprosocial2'] = Video(TR=230, fps=23.98)
od['ses-04_run-01_order-02_content-gockskumara']    = Video(TR= 390, fps=25)

# %%

for video, (TR_LENGTH, fps) in od.items():
    print(video, TR_LENGTH, fps)
    moten = np.load(join(MOTEN_STEP1_DIR, f'moten_video-{video}.npy'))
    # step 1: log transform 
    print(f"* values smaller than 0: {np.sum(moten < 0)}, \n* moten size: {moten.size}, \n* proportion of negative moten values: {np.sum(moten<0)/moten.size}")
    # given that we have few negative numbers, we'll just convert them to 0 
    moten_log = np.nan_to_num(np.log(moten))
    print(f"* mean:{np.mean(moten_log)}, \n* max: {np.max(moten_log)}, \n* min: {np.min(moten_log)}, \n* std: {np.std(moten_log)}")
    print(f"* mean axis 0: {np.mean(moten_log, axis=0)}, \n* std axis 0: {np.std(moten_log, axis = 0)}")
    TR = 0.46
    fmri_rate = 1/TR

    number_of_samples = TR_LENGTH #np.ceil(moten_log.shape[0] * fmri_rate / fps)
    resample_scipy = signal.resample(moten_log, int(number_of_samples), axis = 0)
    print(f"scipyresample shape: {resample_scipy.shape}")
    assert resample_scipy.shape[0] == number_of_samples #TR_LENGTH
    np.save(os.path.join(MOTEN_STEP2_DIR, f'{video}_feature-moten.npy'), resample_scipy)
    # _____________________________________________________________________________
    # resample.append(resample_scipy)
# %%
# resample = []
# for run in [1,2,3,4]:
#     print(f"----------------run: {run}--------------------")
#     moten = np.load(join(moten_dir, f'moten_run-{run}.npy'))


    # # step 1: log transform 
    # print(f"* values smaller than 0: {np.sum(moten < 0)}, \n* moten size: {moten.size}, \n* proportion of negative moten values: {np.sum(moten<0)/moten.size}")
    # # given that we have few negative numbers, we'll just convert them to 0 
    # moten_log = np.nan_to_num(np.log(moten))
    # print(f"* mean:{np.mean(moten_log)}, \n* max: {np.max(moten_log)}, \n* min: {np.min(moten_log)}, \n* std: {np.std(moten_log)}")
    # print(f"* mean axis 0: {np.mean(moten_log, axis=0)}, \n* std axis 0: {np.std(moten_log, axis = 0)}")
    # # step 2: down sample
    # # method 1, scipy _________________________________________________________________
    # fps = 30
    # TR = 2.5
    # fmri_rate = 1/TR

    # number_of_samples = np.ceil(moten_log.shape[0] * fmri_rate / fps)
    # resample_scipy = signal.resample(moten_log, int(number_of_samples), axis = 0)
    # print(f"scipyresample shape: {resample_scipy.shape}")
    # np.save(os.path.join(moten_dir, f'resample-scipy_run-{vi}.npy'), resample_scipy)
    # # _____________________________________________________________________________
    # # resample.append(resample_scipy)

    


# %%
# resample_all = np.vstack(resample)
# model_durs = {1: 369, 2: 341, 3: 372, 4: 406}
# # assert np.sum(model_durs.values()) == resample_all.shape[0]

# # NOTE: The last run (run-4) has one more TR. 
# # We're explicitly removing a TR, due to rounding
# resample_remedy = resample_all[:-1]
# assert np.sum(list(model_durs.values())) == resample_remedy.shape[0]

# np.save(os.path.join(model_dir, 'visual_moten.npy'), resample_remedy)
# %%
