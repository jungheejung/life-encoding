#!/usr/bin/env python

# load packages ________________________________________________________________________________
import pickle
from tikreg import spatial_priors, temporal_priors
from tikreg import models, utils as tikutils
from tikreg import utils as tikutils
from tikreg import models
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
import os, sys, shutil, time, csv
import subprocess
from scipy.io import wavfile
from scipy import stats
import pandas as pd
import numpy as np
import mvpa2.suite as mv



# directories ________________________________________________________________________________
if not os.path.exists('/scratch/f0042x1'):
    os.makedirs('/scratch/f0042x1')
mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
sam_data_dir = '/idata/DBIC/snastase/life'
ridge_dir = '/idata/DBIC/cara/life/ridge'
cara_data_dir = '/idata/DBIC/cara/life/data'
npy_dir = '/idata/DBIC/cara/w2v/w2v_features'
scratch_dir = '/scratch/f0042x1'

participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']

tr_movie = {1: 369, 2: 341, 3: 372, 4: 406}
tr_fmri = {1: 374, 2: 346, 3: 377, 4: 412}
tr_length = 2.5
n_samples = 1509
n_vertices = 40962
n_proc = 32     # how many cores do we have?
n_medial = {'lh': 3486, 'rh': 3491}


# 1. parameters from JOBSUBMIT script  ___________________________________________________________________
model = sys.argv[1]
align = sys.argv[2]
stimfile1 = sys.argv[3]
stimfile2 = sys.argv[4]
stimfile3 = sys.argv[5]
fold = int(sys.argv[6])
fold_shifted = fold + 1

hemi = sys.argv[7]
test_p = sys.argv[8]
# participant = []
# participant.append(part)

included = [1, 2, 3, 4]
included.remove(fold_shifted)
# functions from Cara Van Uden Ridge Regression  ____________________________________________________________________
def get_visual_stim_for_fold(stimfile, fold_shifted, included):
    cam = np.load(os.path.join(npy_dir, '{0}.npy'.format(stimfile)))

    full_stim = []
    full_stim.append(cam[:369, :])
    full_stim.append(cam[369:710, :])
    full_stim.append(cam[710:1082, :])
    full_stim.append(cam[1082:, :])

    for i in range(len(full_stim)):
        this = full_stim[i]
        full_stim[i] = np.concatenate(
            (this[3:, :], this[2:-1, :], this[1:-2, :], this[:-3, :]), axis=1)

    train_stim = [full_stim[i] for i in np.subtract(included, 1)]
    test_stim = full_stim[fold_shifted - 1]

    return train_stim, test_stim


def get_mel():
    mel_list = [[], [], [], []]
    directory = os.path.join(cara_data_dir, 'spectral', 'complete')
    for f in os.listdir(directory):
        if 'csv' in f:
            run = int(f[-5])
            s = pd.read_csv(os.path.join(directory, f))
            filter_col = [col for col in s if col.startswith('mel')]
            tr_s = np.array(s[filter_col])
            tr_avg = np.mean(tr_s, axis=1)

            groupby = tr_avg.shape[0] / tr_movie[run]
            remainder = tr_avg.shape[0] % tr_movie[run]
            tr_reshaped = np.reshape(
                tr_avg[:-remainder], (tr_movie[run], groupby))
            avg = np.mean(tr_reshaped, axis=1)
            mel_list[run - 1] = avg
    return mel_list


def get_ws_data(test_p, fold_shifted, included, hemi):
    print(
        '\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        if run == 4:
            resp = mv.gifti_dataset(os.path.join(
                sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[run], run, hemi))).samples[4:-5, :]
        else:
            resp = mv.gifti_dataset(os.path.join(
                sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[run], run, hemi))).samples[4:-4, :]

        resp = resp[:, cortical_vertices[hemi] == 1]
        mv.zscore(resp, chunks_attr=None)
        print('train', run, resp.shape)

        train_resp.append(resp)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, :]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, :]

    test_resp = test_resp[:, cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)
    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def get_aa_data(test_p, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print(
        '\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]

            resp = resp[:, cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, :]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, :]

    test_resp = test_resp[:, cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def get_ha_common_data(test_p, mappers, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print(
        '\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]

            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            resp = resp[:, cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, :]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, :]

    mv.zscore(test_resp, chunks_attr=None)
    test_resp = mappers[participant].forward(test_resp)
    test_resp = test_resp[:, cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def get_ha_testsubj_data(test_p, mappers, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print(
        '\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]

            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            mv.zscore(resp, chunks_attr=None)
            resp = mappers[test_p].reverse(resp)
            resp = resp[:, cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, cortical_vertices[hemi] == 1]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print proc_stdout


# 2. Load data _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    # 2-1) First let's create mask of cortical vertices excluding medial wall __
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_ds = mv.niml.read(
        '/idata/DBIC/cara/life/ridge/models/niml/ws.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(
        test_ds.samples[1:, :] != 0, axis=0) == 0] = 0

print('Model: {0}\nStim file: {1}, {2}\nHemi: {3}\nRuns in training: {4}\nRun in test: {5}\nParticipant: {6}'.format(
    model, stimfile1, stimfile2, hemi, included, fold_shifted, test_p))

# 2-2) load visual or narrative feature data _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
if model == 'visual':
    # stimfile1 = 'bg'
    X1train_stim, X1test_stim = get_visual_stim_for_fold(
        '{0}_{1}'.format(model, stimfile1), fold_shifted, included)
    # stimfile2 = 'actions'
    X2train_stim, X2test_stim = get_visual_stim_for_fold(
        '{0}_{1}'.format(model, stimfile2), fold_shifted, included)
    # stimfile1 = 'agents'
    X3train_stim, X3test_stim = get_visual_stim_for_fold(
        '{0}_{1}'.format(model, stimfile3), fold_shifted, included)
else:
    # stimfile1 = 'bg'
    X1train_stim, X1test_stim = get_narrative_stim_for_fold(
        '{0}_{1}'.format(model, stimfile1), fold_shifted, included)
    # stimfile2 = 'actions'
    X2train_stim, X2test_stim = get_narrative_stim_for_fold(
        '{0}_{1}'.format(model, stimfile2), fold_shifted, included)
    # stimfile1 = 'agents'
    X3train_stim, X3test_stim = get_narrative_stim_for_fold(
        '{0}_{1}'.format(model, stimfile3), fold_shifted, included)

    # 2-3) load fMRI data __________________________________________________
if align == 'ws':
    Ytrain_unconcat, Ytest = get_ws_data(
        test_p, fold_shifted, included, hemi)
elif align == 'aa':
    Ytrain_unconcat, Ytest = get_aa_data(
        test_p, fold_shifted, included, hemi)
else:
    print('\nLoading hyperaligned mappers...')
    mappers = mv.h5load(os.path.join(
        mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(hemi, fold_shifted)))
    if align == 'ha_common':
        Ytrain_unconcat, Ytest = get_ha_common_data(
            test_p, mappers, fold_shifted, included, hemi)
    elif align == 'ha_testsubj':
        Ytrain_unconcat, Ytest = get_ha_common_data(
            test_p, mappers, fold_shifted, included, hemi)



# 2-4) concatenate 3 runs ______________________________________________
X1train = np.concatenate(X1train_stim)
X2train = np.concatenate(X2train_stim)
X3train = np.concatenate(X3train_stim)
Ytrain = np.concatenate(Ytrain_unconcat)


# load numpy primal weights
# /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded_cara-alpha_loro
result_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/banded-ridge_alpha-cara_loro'
# '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/banded-ridge_alpha-cara_loro/aa/visual/bg_actions_agents/leftout_run_1/sub-rid000001/rh'
# 'kernel-weights_sub-rid000001_model-visual_align-aa_feature-actions_foldshifted-1_hemi_rh.npy'
filename1 = os.path.join(result_dir, align, 'visual', 'bg_actions_agents', 'leftout_run_'+str(fold_shifted), test_p, hemi,
'kernel-weights_'+test_p+'_model-'+model+'_align-'+align+'_feature-'+stimfile1+'_foldshifted-'+str(fold_shifted)+'_hemi_'+hemi+'.npy')
filename2 = os.path.join(result_dir, align, 'visual', 'bg_actions_agents', 'leftout_run_'+str(fold_shifted), test_p, hemi,
'kernel-weights_'+test_p+'_model-'+model+'_align-'+align+'_feature-'+stimfile2+'_foldshifted-'+str(fold_shifted)+'_hemi_'+hemi+'.npy')
filename3 = os.path.join(result_dir, align, 'visual', 'bg_actions_agents', 'leftout_run_'+str(fold_shifted), test_p, hemi,
'kernel-weights_'+test_p+'_model-'+model+'_align-'+align+'_feature-'+stimfile3+'_foldshifted-'+str(fold_shifted)+'_hemi_'+hemi+'.npy')

weights_x1 = np.load(filename1)
weights_x2 = np.load(filename2)
weights_x3 = np.load(filename3)


# load actual
actual_df = pd.DataFrame(data=Ytest)
# vstack weights, hstack test_stim
weights_joint = np.vstack((weights_x1, weights_x2, weights_x3))
teststim_joint = np.hstack((X1test_stim, X2test_stim, X3test_stim))

estimated_ytotal = np.linalg.multi_dot([teststim_joint, weights_joint])

estimated_ytotal_df = pd.DataFrame(data=estimated_ytotal)
corr_total = pd.DataFrame.corrwith(
    estimated_ytotal_df, actual_df, axis=0, method='pearson')

# save as numpy and niml
med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]

# outtotal = np.zeros(
#     (corr_shape + med_wall_ind.shape[0]), dtype=np.dtype(corr_total).type)
# outtotal[selected_node] = corr_total
# np.save(os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-total_foldshifted-{3}_hemi-{4}_range-{5}.npy'.format(
#         test_p, model, align, fold_shifted, hemi, save_nodename)), outtotal)

outtotal = np.zeros(
    (corr_total.shape[0] + med_wall_ind.shape[0]), dtype=np.dtype(corr_total).type)
outtotal[cortical_vertices[hemi] == 1] = corr_total
mv.niml.write(os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-total_foldshifted-{3}_hemi_{4}.niml.dset'.format(
    test_p, model, align, fold_shifted, hemi)), outtotal[None, :])
## add medial wall back in
# copy files and remove files _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
subprocess_cmd('cp -rf /scratch/f0042x1/banded-ridge_alpha-cara_loro/ /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results; rm -rf /scratch/f0042x1/*')
