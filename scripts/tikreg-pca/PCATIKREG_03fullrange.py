#!/usr/bin/env python

# load packages ________________________________________________________________________________
import pickle
from tikreg import spatial_priors, temporal_priors
from tikreg import models, utils as tikutils
from tikreg import utils as tikutils
from tikreg import models
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
import os,sys,shutil,time
import csv
import math
import subprocess
from scipy.io import wavfile
from scipy import stats
import pandas as pd
import numpy as np
import mvpa2.suite as mv
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import json

# directories _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
if not os.path.exists('/scratch/f0042x1'):
    os.makedirs('/scratch/f0042x1')
mvpa_dir = '/dartfs/rc/lab/D/DBIC/DBIC/life_data/hyperalign_mapper' # '/idata/DBIC/cara/life/pymvpa/'
sam_data_dir = '/dartfs/rc/lab/D/DBIC/DBIC/life_data/life_dataset' # '/idata/DBIC/snastase/life'
#ridge_dir = '/idata/DBIC/cara/life/ridge'
#cara_data_dir = '/idata/DBIC/cara/life/data'
npy_dir = '/dartfs/rc/lab/D/DBIC/DBIC/life_data/w2v_feature' # '/idata/DBIC/cara/w2v/w2v_features'
scratch_dir = '/scratch/f0042x1'

participants = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006', 'sub-rid000009', 'sub-rid000012',
                'sub-rid000014', 'sub-rid000017', 'sub-rid000019', 'sub-rid000024', 'sub-rid000027',
                'sub-rid000031', 'sub-rid000032', 'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
                'sub-rid000037', 'sub-rid000038', 'sub-rid000041']

tr_movie = {1: 369, 2: 341, 3: 372, 4: 406}
tr_fmri = {1: 374, 2: 346, 3: 377, 4: 412}
tr_length = 2.5
n_samples = 1509
n_vertices = 40962
n_proc = 32     # how many cores do we have?
n_medial = {'lh': 3486, 'rh': 3491}
increment = 20# 8 hr instead of 30 min


# 1. parameters from JOBSUBMIT script  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
model = sys.argv[1]
align = sys.argv[2]
stimfile1 = sys.argv[3]
stimfile2 = sys.argv[4]
stimfile3 = sys.argv[5]
fold = int(sys.argv[6])
hemi = sys.argv[7]
test_p = sys.argv[8]
start_node = int(sys.argv[9])
# participant = []
# participant.append(part)
fold_shifted = fold + 1
included = [1, 2, 3, 4]
included.remove(fold_shifted)

pca_or_not = True
n_components = 30
pca_dim = 1


# 2-1) First let's create mask of cortical vertices excluding medial wall __
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_ds = mv.niml.read(
        '/idata/DBIC/cara/life/ridge/models/niml/ws.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(
        test_ds.samples[1:, :] != 0, axis=0) == 0] = 0
print("\n1. analysis parameters")
print('Model: {0}\nStim file: {1}, {2}, {3}\nHemi: {3}\nRuns in training: {4}\nRun in test: {5}\nParticipant: {6}'.format(
    model, stimfile1, stimfile2, stimfile3, hemi, included, fold_shifted, test_p))

#nonmedial = cortical_vertices[hemi] == 1

nonmedial = np.where(cortical_vertices[hemi] == 1)[0]
medial = np.where(cortical_vertices[hemi] == 0)[0]

if start_node == round(n_vertices/increment )  :
    node_range = np.arange((start_node-1)*increment, n_vertices)
#    node_range = list(range(40901, 40962))
else:
    node_range = np.arange((start_node-1)*increment, start_node*increment)
#    node_range = list(range((start_node-1)*100+1, start_node*100))

#save_nodename = (start_node-1)*100+1
node_start = node_range[0]
node_end = node_range[-1]
# selected_node = node_range
selected_node = np.intersect1d(nonmedial, node_range)
medial_node = np.intersect1d(medial, node_range)
print("2. node - medial & nonmedial")
print("node range: {0}-{1}".format(node_start, node_end))
print("node shape: {0}".format(selected_node.shape))
# print(type(nonmedial))
# print(nonmedial.shape, "nonmedial")
# print(type(node_range))
# print(node_range.shape, "noderange")
# print(type(selected_node))
# print(selected_node.shape, "selected_node") # ((1073, 3), 'Ytrain')


# ridge_or_not = True
# functions from Cara Van Uden Ridge Regression  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

def PCA_analysis(train_stim, test_stim, n_components, pca_dim=0):
    assert pca_dim == 0 or pca_dim == 1, 'pca_dim should be 0 or 1, but got {}'.format(
        pca_dim)
    if pca_dim == 1:
        train_stim = train_stim.T
        test_stim = test_stim.T

    scaler1 = StandardScaler()
    scaler1.fit(train_stim)

    scaler2 = StandardScaler()
    scaler2.fit(test_stim)

    train_stim = scaler1.transform(train_stim)
    test_stim = scaler2.transform(test_stim)

    if pca_dim == 1:
        train_stim = train_stim.T
        test_stim = test_stim.T

    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(train_stim)

    train_stim = pca.transform(train_stim)
    test_stim = pca.transform(test_stim)

    train_stim = np.array(train_stim)
    test_stim = np.array(test_stim)

    return train_stim, test_stim


def get_visual_stim_for_fold(stimfile, fold_shifted, included):
    pca_or_not = True
    cam = np.load(os.path.join(npy_dir, '{0}.npy'.format(stimfile)))
    full_stim = []
    full_stim.append(cam[:369, :])
    full_stim.append(cam[369:710, :])
    full_stim.append(cam[710:1082, :])
    full_stim.append(cam[1082:, :])

    train_stim = [full_stim[i] for i in np.subtract(included, 1)]
    test_stim = full_stim[fold_shifted - 1]
    print("\n3. visual stim details")
    print("full_stim dim:", len(test_stim))
    len_train_stim = [len(full_stim[i]) for i in np.subtract(included, 1)]
    if pca_or_not:
        step = 300
        train_stim_pca = []
        test_stim_pca = []
        train_stim = np.concatenate(train_stim, axis=0)

        for i in range(0, len(train_stim[0]), step):
            train_temp, test_temp = PCA_analysis(
                train_stim[:, i:i + step], test_stim[:, i:i + step], n_components, pca_dim)
            train_stim_pca.append(
                train_temp), test_stim_pca.append(test_temp)

        train_stim_pca = np.hstack(np.array(train_stim_pca))
        test_stim_pca = np.hstack(np.array(test_stim_pca))

    train_stim_split = []
    cum = 0
    for len_ in len_train_stim:
        train_stim_split.append(train_stim_pca[cum:cum + len_])
        cum += len_
    train_stim_pca = train_stim_split

    for i in range(len(train_stim_pca)):
        this = train_stim_pca[i]
        train_stim_pca[i] = np.concatenate(
            (this[3:, :], this[2:-1, :], this[1:-2, :], this[:-3, :]), axis=1)

    test_stim_pca = np.concatenate(
        (test_stim_pca[3:, :], test_stim_pca[2:-1, :], test_stim_pca[1:-2, :], test_stim_pca[:-3, :]), axis=1)

    return train_stim_pca, test_stim_pca


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
    print("4. within subject data")
    print(
        'Loading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        if run == 4:
            resp = mv.gifti_dataset(os.path.join(
                sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[run], run, hemi))).samples[4:-5, :]
        else:
            resp = mv.gifti_dataset(os.path.join(
                sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[run], run, hemi))).samples[4:-4, :]

        # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
        # DELETE LATER -
        # resp = resp[:, c##ortical_verticee[hemi] == 1]

        resp = resp[:, selected_node]
        # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
        mv.zscore(resp, chunks_attr=None)


        print('train', run, resp.shape)

        train_resp.append(resp)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, :]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, :]

    # test_resp = test_resp[:, cortical_verticee[hemi] == 1]
    test_resp = test_resp[:, selected_node]
    mv.zscore(test_resp, chunks_attr=None)
    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def get_aa_data(test_p, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print("4. anatomical alignment data")
    print(
        'Loading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # UNCOMMENT LATER -
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # DELETE LATER -
            # resp = resp[:, cortical_vertices[hemi] == 1]

            resp = resp[:, selected_node]
            mv.zscore(resp, chunks_attr=None)
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

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
    # test_resp = test_resp[:, cortical_vertices[hemi] == 1]
    test_resp = test_resp[:, selected_node]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def get_ha_common_data(test_p, mappers, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print("\n4. hyperalignment common data")
    print(
        'Loading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # UNCOMMENT LATER -
            # if run == 4:
            #     resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            #         participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            # else:
            #     resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            #         participant, tr_fmri[run], run, hemi))).samples[4:-4, :]
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]
            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # DELETE LATER -
            # resp = resp[:, cortical_vertices[hemi] == 1]

            resp = resp[:, selected_node]
            #
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
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
    # test_resp = test_resp[:, cortical_vertices[hemi] == 1]
    test_resp = test_resp[:, selected_node]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def get_ha_testsubj_data(test_p, mappers, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print("\n4. hyperalignment test subject data")
    print(
        'Loading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # UNCOMMENT LATER -
            # if run == 4:
            #     resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            #         participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            # else:
            #     resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            #         participant, tr_fmri[run], run, hemi))).samples[4:-4, :]
            # # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # DELETE LATER -
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            mv.zscore(resp, chunks_attr=None)
            resp = mappers[test_p].reverse(resp)
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # DELETE LATER -
            # resp = resp[:, cortical_vertices[hemi] == 1]

            resp = resp[:, selected_node]
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            mv.zscore(resp, chunks_attr=None)


            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    # if fold_shifted == 4:
    #     test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
    #         test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, cortical_vertices[hemi] == 1]
    # else:
    #     test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
    #         test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, cortical_vertices[hemi] == 1]

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, selected_node]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, selected_node]

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
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)


# 2. Load data _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

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

# delete later: ONLY ONE NODE
# print(Ytrain_unconcat.shape, "Y train unconcatenate shape")

# 2-4) concatenate 3 runs ______________________________________________
X1train = np.concatenate(X1train_stim)
X2train = np.concatenate(X2train_stim)
X3train = np.concatenate(X3train_stim)
Ytrain = np.concatenate(Ytrain_unconcat)

# 2-5) Print for JOB LOG ___________________________________________________
print('\nShape of training and testing set')
print(X1train.shape, "X1train shape") # ((1073, 120), 'X1train')
print(X2train.shape, "X2train shape") # ((1073, 120), 'X2train')
print(X3train.shape, "X3train shape") # ((1073, 120), 'X3train')
print(X1test_stim.shape, "X1test stim") # ((403, 120), 'X1test_stim')
print(X2test_stim.shape, "X2test stim") # ((403, 120), 'X2test_stim')
print(X3test_stim.shape, "X3test stim") # ((403, 120), 'X3test_stim')
print(Ytest.shape, "Ytest") # ((403, 3), 'Ytest')
print(Ytrain.shape, "Ytrain") # ((1073, 3), 'Ytrain')


# 3. [ banded ridge ] alpha and ratios _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

alphas = np.logspace(0, 3, 20)  # commented out for current analysis
# alphas = [18.33] # currently using for our quick analysis. For the main analysis, use the full logspace as above
ratios = np.logspace(-2, 2, 25)
print("\nalphas: {0}".format(alphas))
print("\nRatios: {0}".format(ratios))

train_id = np.arange(X1train.shape[0])
dur1, dur2, dur3 = tr_movie[included[0]] - \
    3, tr_movie[included[1]] - 3, tr_movie[included[2]] - 3


# 4. [ banded ridge ] setting up loro and priors _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

loro1 = [(train_id[:dur1 + dur2], train_id[dur1 + dur2:]),
         (np.concatenate((train_id[:dur1], train_id[dur1 + dur2:]), axis=0),
          train_id[dur1:dur1 + dur2]),
         (train_id[dur1:], train_id[:dur1])]

X1_prior = spatial_priors.SphericalPrior(X1train, hyparams=ratios)
X2_prior = spatial_priors.SphericalPrior(X2train, hyparams=ratios)
X3_prior = spatial_priors.SphericalPrior(X3train, hyparams=ratios)

temporal_prior = temporal_priors.SphericalPrior(delays=[0])  # no delays


# 5. [ banded ridge ] banded ridge regression _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

fit_banded_polar = models.estimate_stem_wmvnp([X1train, X2train, X3train], Ytrain,
                                              [X1test_stim, X2test_stim,
                                                  X3test_stim], Ytest,
                                              feature_priors=[
                                                  X1_prior, X2_prior, X3_prior],
                                              temporal_prior=temporal_prior,
                                              ridges=alphas, folds=loro1,
                                              performance=True, weights=True, verbosity=False)

voxelwise_optimal_hyperparameters = fit_banded_polar['optima']
print('\nVoxelwise optimal hyperparameter shape: {0}'.format(voxelwise_optimal_hyperparameters.shape))

# Faster conversion of kernel weights to primal weights via matrix multiplication
# each vector (new_alphas, lamda_ones, lamda_twos) contains v number of entries (e.g. voxels)
new_alphas = voxelwise_optimal_hyperparameters[:, -1]
lambda_ones = voxelwise_optimal_hyperparameters[:, 1]
lambda_twos = voxelwise_optimal_hyperparameters[:, 2]
lambda_threes = voxelwise_optimal_hyperparameters[:, 3]


# 6. [ banded ridge ] calculating primal weights from kernel weights _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

kernel_weights = fit_banded_polar['weights']
# 6-1. calculate the beta weights from primal weights ______________________
weights_x1 = np.linalg.multi_dot(
    [X1train.T, kernel_weights, np.diag(new_alphas), np.diag(lambda_ones**-2)])
weights_x2 = np.linalg.multi_dot(
    [X2train.T, kernel_weights, np.diag(new_alphas), np.diag(lambda_twos**-2)])
weights_x3 = np.linalg.multi_dot(
    [X3train.T, kernel_weights, np.diag(new_alphas), np.diag(lambda_threes**-2)])

weights_joint = np.vstack((weights_x1, weights_x2, weights_x3))
teststim_joint = np.hstack((X1test_stim, X2test_stim, X3test_stim))
print("Feature 1 weight shape: {0}".format(weights_x1.shape)) #(120, 1)
print("Feature 2 weight shape: {0}".format(weights_x2.shape)) #(120, 1)
print("Feature 3 weight shape: {0}".format(weights_x3.shape)) #(120, 1)
print("weights type: {0}".format(type(weights_x1)))
print("Joint weights shape: {0}".format(weights_joint.shape))
print("Joint stim shape: {0}".format(teststim_joint.shape))

# 6-2. calculate the estimated Y based on the primal weights _________________
estimated_y1 = np.linalg.multi_dot([X1test_stim, weights_x1]) # (369, 120) * (120, 5)
estimated_y2 = np.linalg.multi_dot([X2test_stim, weights_x2])
estimated_y3 = np.linalg.multi_dot([X3test_stim, weights_x3])
estimated_ytotal = np.linalg.multi_dot([teststim_joint, weights_joint]) # (369, 360) * (360, 5)

directory = os.path.join(scratch_dir, 'PCA_tikreg-loro_fullrange-chunk',
                         '{0}/{1}/{2}_{3}_{4}/leftout_run_{5}'.format(align, model, stimfile1, stimfile2, stimfile3, fold_shifted), test_p, hemi)
if not os.path.exists(directory):
    os.makedirs(directory)

print("\nsave directory: {0}".format(directory))

print("\nalpha shape: {0}".format(new_alphas.shape))
print("alpha type: {0}".format(type(new_alphas)))

# 6-4. save alpha_________________________________________________
# corr_shape = n_vertices - n_medial[hemi]
# outhyperparam = np.zeros(
#     (corr_shape + med_wall_ind.shape[0]), dtype=new_alphas.dtype)
#outhyperparam = np.zeros(node_range.shape[0], dtype=new_alphas.dtype)
print("\n6-4. alphas")

ind_nonmedial = np.array(selected_node) # insert nonmedial index
print("ind_nonmedial: {0}".format(ind_nonmedial))
ind_medial = np.array(medial_node) # insert medial index
print("ind_medial: {0}".format(ind_medial))
append_zero = np.zeros(len(medial_node)) # insert medial = 0
alpha_nonmedial = np.array(new_alphas) # insert nonmedial alpha
print("append_zero: {0}".format(append_zero))
print("alpha_nonmedial: {0}".format(alpha_nonmedial))
weight_x1_nonmedial = np.array(weights_x1)
weight_x2_nonmedial = np.array(weights_x2)
weight_x3_nonmedial = np.array(weights_x3) # (120, 5)
weights_joint_nonmedial = np.array(weights_joint)

index_chunk = np.concatenate((ind_nonmedial,ind_medial), axis = None)
alpha_value = np.concatenate((alpha_nonmedial,append_zero),axis = None)

zipped_alphas = zip(index_chunk.astype(float), alpha_value.astype(float))
sorted_alphas = sorted(zipped_alphas)


#outhyperparam[outhyperparam] = new_alphas

# 6-4. save alpha
# https://stackoverflow.com/questions/64082441/save-a-list-of-tuples-to-a-file-and-read-it-again-as-a-list
alpha_savename = os.path.join(directory, 'hyperparam-alpha_{0}_model-{1}_align-{2}_foldshifted-{3}_hemi-{4}_range-{5}-{6}.json'.format(
        test_p, model, align,  fold_shifted, hemi,node_start, node_end))
with open(alpha_savename, 'w') as f:
     json.dump(sorted_alphas, f)

# 6-3. save primal weights _________________________________________________
# [ ] TO DO: primal weights. make sure to grab the shape and create numpy zeros of that shape
# [ ] save tuple index and numpy array
print("\n6-3.save primal weights")
print("weights nonmedial shape: {0}".format(weight_x3_nonmedial.shape[0])) #(120, 5)
print("length of medial nodes: {0}".format(len(medial_node)))

if len(medial_node) != 0:
    index_chunk = np.concatenate((ind_nonmedial,ind_medial), axis = None)
    weight_zero = np.zeros((weight_x3_nonmedial.shape[0], len(medial_node))) # 120, 6# insert medial = 0
    weight_j = np.zeros((weight_x3_nonmedial.shape[0]*3, len(medial_node)))
    print("weight_zero shape: {0}".format(weight_x3_nonmedial.shape)) # 120, 14
    weightx1_value = np.transpose(np.hstack((weight_x1_nonmedial,weight_zero)))
    weightx2_value = np.transpose(np.hstack((weight_x2_nonmedial,weight_zero)))
    weightx3_value = np.transpose(np.hstack((weight_x3_nonmedial,weight_zero)))
    weightj_value  = np.transpose(np.hstack((weights_joint_nonmedial,weight_zero))) #360, 14, 360, 5
    print("weightx1_value shape: {0}".format(weightx1_value.shape)) # 600
elif len(medial_node) == 0:
    index_chunk    = ind_nonmedial
    weightx1_value = np.transpose(weight_x1_nonmedial)
    weightx2_value = np.transpose(weight_x2_nonmedial)
    weightx3_value = np.transpose(weight_x3_nonmedial)
    weightj_value  = np.transpose(weights_joint_nonmedial)
    print("weightx1_value shape: {0}".format(weightx1_value.shape))

#numpy transpose weight_x3_nonmedial (120, 5) -> (5, 120)
w_x1_dict = {e: weightx1_value[i] for i, e in enumerate(index_chunk)}
w_x2_dict = {e: weightx2_value[i] for i, e in enumerate(index_chunk)}
w_x3_dict = {e: weightx3_value[i] for i, e in enumerate(index_chunk)}
w_xj_dict = {e: weightj_value[i] for i, e in enumerate(index_chunk)}

zipped_weightx1 = zip(index_chunk.astype(float), weightx1_value.astype(float))
sorted_weightx1 = sorted(zipped_weightx1)
zipped_weightx2 = zip(index_chunk.astype(float), weightx2_value.astype(float))
sorted_weightx2 = sorted(zipped_weightx2)
zipped_weightx3 = zip(index_chunk.astype(float), weightx3_value.astype(float))
sorted_weightx3 = sorted(zipped_weightx3)
zipped_weightj = zip(index_chunk.astype(float), weightj_value.astype(float))
sorted_weightj = sorted(zipped_weightj)

weightx1_savename = os.path.join(directory, 'primal-weights_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}-{7}.json'.format(
        test_p, model, align, stimfile1, fold_shifted, hemi,node_start, node_end))
weightx2_savename = os.path.join(directory, 'primal-weights_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}-{7}.json'.format(
        test_p, model, align, stimfile2, fold_shifted, hemi,node_start, node_end))
weightx3_savename = os.path.join(directory, 'primal-weights_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}-{7}.json'.format(
        test_p, model, align, stimfile3, fold_shifted, hemi,node_start, node_end))
weightj_savename = os.path.join(directory, 'primal-weights_{0}_model-{1}_align-{2}_feature-total_foldshifted-{3}_hemi-{4}_range-{5}-{6}.json'.format(
        test_p, model, align, fold_shifted, hemi, node_start, node_end))

with open(weightx1_savename, 'w') as f:
     json.dump(sorted_weightx1, f)
with open(weightx2_savename, 'w') as f:
     json.dump(sorted_weightx2, f)
with open(weightx3_savename, 'w') as f:
     json.dump(sorted_weightx3, f)
with open(weightj_savename, 'w') as f:
     json.dump(sorted_weightj, f)

# 7. [ banded ridge ] correlation coefficient between actual Y and estimated Y _ _ _ _ _ _ _

actual_df = pd.DataFrame(data=Ytest)
estimated_y1_df = pd.DataFrame(data=estimated_y1)
estimated_y2_df = pd.DataFrame(data=estimated_y2)
estimated_y3_df = pd.DataFrame(data=estimated_y3)
estimated_ytotal_df = pd.DataFrame(data=estimated_ytotal)

corr_x1 = pd.DataFrame.corrwith(
    estimated_y1_df, actual_df, axis=0, method='pearson')
corr_x2 = pd.DataFrame.corrwith(
    estimated_y2_df, actual_df, axis=0, method='pearson')
corr_x3 = pd.DataFrame.corrwith(
    estimated_y3_df, actual_df, axis=0, method='pearson')
corr_total = pd.DataFrame.corrwith(
    estimated_ytotal_df, actual_df, axis=0, method='pearson')

# 7-1. save files _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
print("7. correlation values")
corr_x1_nonmedial = corr_x1.to_numpy()
corr_x2_nonmedial = corr_x2.to_numpy()
corr_x3_nonmedial = corr_x3.to_numpy()
corr_t_nonmedial = corr_total.to_numpy()

if len(medial_node) != 0:
    index_chunk = np.concatenate((ind_nonmedial,ind_medial), axis = None)
    weight_zero = np.zeros((weight_x3_nonmedial.shape[0], len(medial_node))) # insert medial = 0
    corrx1_value = np.stack((corr_x1_nonmedial,append_zero))
    corrx2_value = np.stack((corr_x2_nonmedial,append_zero))
    corrx3_value = np.stack((corr_x3_nonmedial,append_zero))
    corr_t_value = np.stack((corr_t_nonmedial,append_zero))
elif len(medial_node) == 0:
    index_chunk = ind_nonmedial
    corrx1_value = corr_x1_nonmedial
    corrx2_value = corr_x2_nonmedial
    corrx3_value = corr_x3_nonmedial
    corr_t_value = corr_t_nonmedial
    print("weightx1_value shape: {0}".format( weightx1_value.shape[0]))

zipped_corrx1 = zip(index_chunk.astype(float), corrx1_value.astype(float))
zipped_corrx2 = zip(index_chunk.astype(float), corrx2_value.astype(float))
zipped_corrx3 = zip(index_chunk.astype(float), corrx3_value.astype(float))
zipped_corrjoint = zip(index_chunk.astype(float), corr_t_value.astype(float))

sorted_corrx1 = sorted(zipped_corrx1);
sorted_corrx2 = sorted(zipped_corrx2);
sorted_corrx3 = sorted(zipped_corrx3);
sorted_corrjoint = sorted(zipped_corrjoint);


corrx1_savename = os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}-{7}.json'.format(
        test_p, model, align, stimfile1, fold_shifted, hemi,node_start, node_end))
corrx2_savename = os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}-{7}.json'.format(
        test_p, model, align, stimfile2, fold_shifted, hemi,node_start, node_end))
corrx3_savename = os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}-{7}.json'.format(
        test_p, model, align, stimfile3, fold_shifted, hemi,node_start, node_end))
corrt_savename = os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-total_foldshifted-{3}_hemi-{4}_range-{5}-{6}.json'.format(
        test_p, model, align, fold_shifted, hemi, node_start, node_end))

with open(corrx1_savename, 'w') as f:
     json.dump(sorted_corrx1, f)
with open(corrx2_savename, 'w') as f:
     json.dump(sorted_corrx2, f)
with open(corrx3_savename, 'w') as f:
     json.dump(sorted_corrx3, f)
with open(corrt_savename, 'w') as f:
     json.dump(sorted_corrjoint, f)

## add medial wall back in
# copy files and remove files _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
subprocess_cmd('cp -rf /scratch/f0042x1/PCA_tikreg-loro_fullrange-chunk/ /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results; rm -rf /scratch/f0042x1/*')

print("\nprocess complete")
