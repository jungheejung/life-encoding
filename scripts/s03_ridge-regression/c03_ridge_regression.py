#!/usr/bin/env python

# load packages ________________________________________________________________________________
import matplotlib; matplotlib.use('agg')
import mvpa2.suite as mv
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import wavfile
from utils import hyper_ridge
import sys, os, time, csv
from sklearn.linear_model import RidgeCV

# directories ________________________________________________________________________________
mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
sam_data_dir = '/idata/DBIC/snastase/life'
ridge_dir = '/idata/DBIC/cara/life/ridge'
cara_data_dir = '/idata/DBIC/cara/life/data'
npy_dir = '/idata/DBIC/cara/w2v/w2v_features'

participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']

hemispheres = ['lh', 'rh']

tr_movie = {1:369, 2:341, 3:372, 4:406}
tr_fmri = {1:374, 2:346, 3:377, 4:412}
tr_length = 2.5
n_samples = 1509
n_vertices = 40962
n_proc = 32     # how many cores do we have?
n_medial = {'lh': 3486, 'rh': 3491}

# functions ____________________________________________________________________
def get_visual_stim_for_fold(stimfile, fold_shifted, included):
    cam = np.load(os.path.join(npy_dir, '{0}.npy'.format(stimfile)))
    # motion = np.load('/ihome/cara/global_motion/motion_downsampled_complete.npy')
    #
    # motion_list = []
    # motion_list.append(motion[:369])
    # motion_list.append(motion[369:710])
    # motion_list.append(motion[710:1082])
    # motion_list.append(motion[1082:])

    full_stim = []
    full_stim.append(cam[:369,:])
    full_stim.append(cam[369:710,:])
    full_stim.append(cam[710:1082,:])
    full_stim.append(cam[1082:,:])

    for i in range(len(full_stim)):
        # m = motion_list[i]
    	# m_avg = np.mean(np.vstack((m[3:], m[2:-1], m[1:-2], m[:-3])),axis=0)
    	# m_avg = np.reshape(m_avg,(-1,1))

        this = full_stim[i]
        # full_stim[i] = np.concatenate((m_avg, this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)
        full_stim[i] = np.concatenate((this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)

    train_stim = [full_stim[i] for i in np.subtract(included, 1)]
    test_stim = full_stim[fold_shifted-1]

    return train_stim, test_stim

def get_mel():
    mel_list = [[],[],[],[]]
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
            tr_reshaped = np.reshape(tr_avg[:-remainder], (tr_movie[run], groupby))
            avg = np.mean(tr_reshaped, axis=1)
            mel_list[run-1] = avg
    return mel_list

def get_sound_envelope():
    audiolist = []

    for run in range(1,5):
        audio_fn = os.path.join('/idata/DBIC/cara/life/data/audio', 'life_part{0}.wav'.format(run))

        # Hemodynamic response function from AFNI
        dt = np.arange(0, 10)
        p = 8.6
        q = 0.547
        hrf = ((dt / (p * q)) ** p) * np.exp(p - dt / q)

        # Load audio file
        frequency, audio = wavfile.read(audio_fn)

        duration = np.round(len(audio) / float(frequency)).astype(int)

        # Compute sound envelope (root mean squared)
        envelope = np.array([np.sqrt(np.mean(t ** 2)) for t in
                             np.array_split(audio, duration)])

        envelope = np.nan_to_num(envelope)
        # Convolve with HRF
        envelope = np.convolve(envelope, hrf)[:duration]

        # Downsample to TR and z-score (and plot)
        envelope = stats.zscore(np.interp(np.linspace(0, (tr_movie[run] - 1) * tr_length, tr_movie[run]),
                                    np.arange(0, duration), envelope))

        audiolist.append(envelope)
    print(audiolist[2])
    return audiolist

def get_narrative_stim_for_fold(stimfile, fold_shifted, included):
    cam = np.load(os.path.join(npy_dir, '{0}.npy'.format(stimfile)))

    # nuisance = get_mel()
    nuisance = get_sound_envelope()

    full_stim = []
    full_stim.append(cam[:369,:])
    full_stim.append(cam[369:710,:])
    full_stim.append(cam[710:1082,:])
    full_stim.append(cam[1082:,:])

    for i in range(len(full_stim)):
        n = nuisance[i]
    	n_avg = np.mean(np.vstack((n[3:], n[2:-1], n[1:-2], n[:-3])),axis=0)
    	n_avg = np.reshape(n_avg,(-1,1))
        this = full_stim[i]

        full_stim[i] = np.concatenate((n_avg, this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)

    train_stim = [full_stim[i] for i in np.subtract(included, 1)]
    test_stim = full_stim[fold_shifted-1]

    return train_stim, test_stim


def get_ws_data(test_p, fold_shifted, included, hemi):
    print('\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        if run == 4:
            resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[run], run, hemi))).samples[4:-5,:]
        else:
            resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[run], run, hemi))).samples[4:-4,:]

        resp = resp[:,cortical_vertices[hemi] == 1]
        mv.zscore(resp, chunks_attr=None)
        print('train', run, resp.shape)

        train_resp.append(resp)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5,:]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4,:]

    test_resp = test_resp[:,cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)
    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp

def get_aa_data(test_p, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print('\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr_fmri[run], run, hemi))).samples[4:-5,:]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr_fmri[run], run, hemi))).samples[4:-4,:]

            resp = resp[:,cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5,:]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4,:]

    test_resp = test_resp[:,cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp

def get_ha_common_data(test_p, mappers, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print('\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr_fmri[run], run, hemi))).samples[4:-5,:]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr_fmri[run], run, hemi))).samples[4:-4,:]

            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            resp = resp[:,cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5,:]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4,:]

    mv.zscore(test_resp, chunks_attr=None)
    test_resp = mappers[participant].forward(test_resp)
    test_resp = test_resp[:,cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp

def get_ha_testsubj_data(test_p, mappers, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print('\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr_fmri[run], run, hemi))).samples[4:-5,:]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr_fmri[run], run, hemi))).samples[4:-4,:]

            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            mv.zscore(resp, chunks_attr=None)
            resp = mappers[test_p].reverse(resp)
            resp = resp[:,cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5,cortical_vertices[hemi] == 1]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4,cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp

# parameters ___________________________________________________________________

# ${MODEL} ${ALIGN} ${STIM} ${RUN} ${HEMI}

model = sys.argv[1]
align = sys.argv[2]
stimfile = sys.argv[3]
fold = int(sys.argv[4])
fold_shifted = fold+1

hemi = sys.argv[5]

included = [1,2,3,4]
included.remove(fold_shifted)

# main code ___________________________________________________________________

# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_ds = mv.niml.read('/idata/DBIC/cara/life/ridge/models/niml/ws.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(test_ds.samples[1:, :] != 0, axis=0) == 0] = 0

print('Model: {0}\nStim file: {1}\nHemi: {2}\nRuns in training: {3}\nRun in test: {4}\n'.format(model, stimfile, hemi, included, fold_shifted))

if model == 'visual':
    train_stim, test_stim = get_visual_stim_for_fold('{0}_{1}'.format(model, stimfile), fold_shifted, included)
else:
    train_stim, test_stim = get_narrative_stim_for_fold('{0}_{1}'.format(model, stimfile), fold_shifted, included)

for test_p in participants:
    if align == 'ws':
        train_resp, test_resp = get_ws_data(test_p, fold_shifted, included, hemi)
    elif align == 'aa':
        train_resp, test_resp = get_aa_data(test_p, fold_shifted, included, hemi)
    else:
        print('\nLoading hyperaligned mappers...')
        mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(hemi, fold_shifted)))
        if align == 'ha_common':
            train_resp, test_resp = get_ha_common_data(test_p, mappers, fold_shifted, included, hemi)
        elif align == 'ha_testsubj':
            train_resp, test_resp = get_ha_testsubj_data(test_p, mappers, fold_shifted, included, hemi)

    alphas = np.logspace(0, 3, 20)
    nboots = len(included)
    chunklen = 15
    nchunks = 15

    wt, corrs, alphas, bootstrap_corrs, valinds = hyper_ridge.bootstrap_ridge(train_stim, train_resp, test_stim, test_resp, alphas, nboots, single_alpha=True, nuisance_regressor=True)

    print('\nFinished training ridge regression')
    print('\n\nwt: {0}'.format(wt))
    print('\n\n corrs: {0}'.format(corrs))
    print('\n\nalphas: {0}'.format(alphas))
    print('\n\nbootstrap_corrs: {0}'.format(bootstrap_corrs))
    print('\n\nvalinds: {0}'.format(valinds))

    print('\n\nWriting to file...')
    directory = os.path.join('/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/ridge-models', '{0}/{1}/{2}/leftout_run_{3}'.format(align, model, stimfile, fold_shifted), test_p, hemi)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save the weights
    np.save(os.path.join(directory, 'weights.npy'), wt)

    # for corrtype in ['full_corrs', 'no_nuisance_corrs', 'nuisance_corrs']:
    for corrtype in ['corrs']:
        # if corrtype == 'full_corrs':
        if corrtype == 'corrs':
            pred = np.dot(test_stim, wt)
        elif corrtype == 'no_nuisance_corrs':
            pred = np.dot(test_stim[:,1:], wt[1:,:])
        elif corrtype == 'nuisance_corrs':
            pred = np.dot(test_stim[:,0][:,None], wt[0,:][None,:])

        # Find prediction correlations
        nnpred = np.nan_to_num(pred)
        corrs = np.nan_to_num(np.array([np.corrcoef(test_resp[:,ii], nnpred[:,ii].ravel())[0,1]
                                            for ii in range(test_resp.shape[1])]))

        # save the corrs
        np.save(os.path.join(directory, '{0}.npy'.format(corrtype)), corrs)
        med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
        out = np.zeros((corrs.shape[0]+med_wall_ind.shape[0]),dtype= corrs.dtype)
        out[cortical_vertices[hemi] == 1] = corrs
        mv.niml.write(os.path.join(directory, '{0}.{1}.niml.dset'.format(corrtype, hemi)), out[None,:])

    # save the alphas, bootstrap corrs, and valinds
    np.save(os.path.join(directory, 'alphas.npy'), alphas)
    np.save(os.path.join(directory, 'bootstrap_corrs.npy'), bootstrap_corrs)
    np.save(os.path.join(directory, 'valinds.npy'), valinds)

    print('\nFinished writing to {0}'.format(directory))
print('all done!')
