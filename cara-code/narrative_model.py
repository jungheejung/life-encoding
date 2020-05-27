# python life.py | tee logs/other_$(date +"%F-T%H%M%S").log
import matplotlib; matplotlib.use('agg')

import mvpa2.suite as mv
import numpy as np
from scipy import stats
from utils import hyper_ridge
import sys, os, time, csv
import pandas as pd

mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
sam_data_dir = '/idata/DBIC/snastase/life'
ridge_dir = '/idata/DBIC/cara/life/ridge'
cara_data_dir = '/idata/DBIC/cara/life/data/'
npy_dir = '/dartfs-hpc/scratch/cara/w2v/w2v_features'


participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']
hemispheres = ['lh', 'rh']

tr = {1:374, 2:346, 3:377, 4:412}
n_samples = 1509
n_vertices = 40962
n_proc = 32     # how many cores do we have?
n_medial = {'lh': 3486, 'rh': 3491}

def get_mel_list():
    mel_list = [[],[],[],[]]
    directory = os.path.join(cara_data_dir, 'spectral', 'complete')
    for f in os.listdir(directory):
        if 'csv' in f:
            print(f)
            run = int(f[-5])
            s = pd.read_csv(os.path.join(directory, f))
            filter_col = [col for col in s if col.startswith('mel')]
            tr_s = np.array(s[filter_col])
            tr_avg = np.mean(tr_s, axis=1)

            print(tr_avg.shape)
            groupby = tr_avg.shape[0] / tr_dict[run]
            remainder = tr_avg.shape[0] % tr_dict[run]
            tr_reshaped = np.reshape(tr_avg[:-remainder], (tr_dict[run], groupby))
            avg = np.mean(tr_reshaped, axis=1)
            print(avg.shape)
            mel_list[run-1] = avg
    return mel_list

def get_stim_for_fold(stimfile, fold_shifted, included):
    cam = np.load(os.path.join(npy_dir, '{0}.npy'.format(stimfile)))

    mel_list = get_mel_list()

    full_stim = []
    full_stim.append(cam[:369,:])
    full_stim.append(cam[369:710,:])
    full_stim.append(cam[710:1082,:])
    full_stim.append(cam[1082:,:])

    for i in range(len(full_stim)):
        m = mel_list[i]
    	m_avg = np.mean(np.vstack((m[3:], m[2:-1], m[1:-2], m[:-3])),axis=0)
    	m_avg = np.reshape(m_avg,(-1,1))

        this = full_stim[i]
        full_stim[i] = np.concatenate((m_avg, this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)
        print(i+1, full_stim[i].shape)

    train_stim = [full_stim[i] for i in np.subtract(included, 1)]
    test_stim = full_stim[fold_shifted-1]

    return train_stim, test_stim

def get_ha_testsubj_data(test_p, mappers, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print('\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[run], run, hemi))).samples[4:-5,:]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[run], run, hemi))).samples[4:-4,:]

            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            mv.zscore(resp, chunks_attr=None)
            resp = mappers[test_p].reverse(resp)
            resp = resp[:,cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr[fold_shifted], fold_shifted, hemi))).samples[4:-5,cortical_vertices[hemi] == 1]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr[fold_shifted], fold_shifted, hemi))).samples[4:-4,cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp

stimfile = sys.argv[1]
fold = int(sys.argv[2])
fold_shifted = fold+1

hemi = sys.argv[3]

included = [1,2,3,4]
included.remove(fold_shifted)

# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_ds = mv.niml.read('/dartfs-hpc/scratch/cara/models/niml/ws/ws_run1_singlealpha.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(test_ds.samples[1:, :] != 0, axis=0) == 0] = 0

print('Stim file: {0}\nHemi: {1}\nRuns in training: {2}\nRun in test: {3}\n'.format(stimfile, hemi, included, fold_shifted))

train_stim, test_stim = get_stim_for_fold(stimfile, fold_shifted, included)

print('\nLoading hyperaligned mappers...')
mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(hemi, fold_shifted)))

for test_p in participants:
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
    directory = os.path.join('/dartfs-hpc/scratch/cara/new_models/narrative', '{0}/run_{1}'.format(stimfile, fold_shifted), test_p, hemi)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save the weights
    np.save(os.path.join(directory, 'weights.npy'), wt)

    w_pred = np.dot(test_stim, wt)
    wo_pred = np.dot(test_stim[:,1:], wt[1:,:])

    # Find prediction correlations
    w_nnpred = np.nan_to_num(w_pred)
    wo_nnpred = np.nan_to_num(wo_pred)

    w_corrs = np.nan_to_num(np.array([np.corrcoef(test_resp[:,ii], w_nnpred[:,ii].ravel())[0,1]
                                        for ii in range(test_resp.shape[1])]))
    wo_corrs = np.nan_to_num(np.array([np.corrcoef(test_resp[:,ii], wo_nnpred[:,ii].ravel())[0,1]
                                        for ii in range(test_resp.shape[1])]))

    # save the corrs with nuisance regressor
    np.save(os.path.join(directory, 'corrs_w_nuisance.npy'), w_corrs)
    med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
    out = np.zeros((w_corrs.shape[0]+med_wall_ind.shape[0]),dtype= w_corrs.dtype)
    out[cortical_vertices[hemi] == 1] = w_corrs
    mv.niml.write(os.path.join(directory, 'corrs_w_nuisance.{0}.niml.dset'.format(hemi)), out[None,:])

    # save the corrs without nuisance regressor
    np.save(os.path.join(directory, 'corrs_wo_nuisance.npy'), wo_corrs)
    med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
    out = np.zeros((wo_corrs.shape[0]+med_wall_ind.shape[0]),dtype= wo_corrs.dtype)
    out[cortical_vertices[hemi] == 1] = wo_corrs
    mv.niml.write(os.path.join(directory, 'wo_corrs.{0}.niml.dset'.format(hemi)), out[None,:])

    # save the alphas, bootstrap corrs, and valinds
    np.save(os.path.join(directory, 'alphas.npy'), alphas)
    np.save(os.path.join(directory, 'bootstrap_corrs.npy'), bootstrap_corrs)
    np.save(os.path.join(directory, 'valinds.npy'), valinds)

    print('\nFinished writing to {0}'.format(directory))
print('all done!')
