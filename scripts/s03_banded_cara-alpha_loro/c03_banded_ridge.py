#!/usr/bin/env python

# load packages ________________________________________________________________________________
import matplotlib; matplotlib.use('agg')
import mvpa2.suite as mv
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import wavfile
import sys, os, time, csv
from sklearn.linear_model import RidgeCV
#from nilearn.plotting import plot_surf
import matplotlib.pyplot as plt
from tikreg import models
from tikreg import utils as tikutils

from tikreg import models, utils as tikutils
from tikreg import spatial_priors, temporal_priors

import pickle
# directories ________________________________________________________________________________
mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
sam_data_dir = '/idata/DBIC/snastase/life'
ridge_dir = '/idata/DBIC/cara/life/ridge'
cara_data_dir = '/idata/DBIC/cara/life/data'
npy_dir = '/idata/DBIC/cara/w2v/w2v_features'

# participants = ['sub-rid000001']
#,'sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
#                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
#                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
#                'sub-rid000037','sub-rid000038','sub-rid000041']

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

    full_stim = []
    full_stim.append(cam[:369,:])
    full_stim.append(cam[369:710,:])
    full_stim.append(cam[710:1082,:])
    full_stim.append(cam[1082:,:])

    for i in range(len(full_stim)):
        this = full_stim[i]
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
model = sys.argv[1]
align = sys.argv[2]
stimfile1 = sys.argv[3]
stimfile2 = sys.argv[4]
fold = int(sys.argv[5])
fold_shifted = fold+1

hemi = sys.argv[6]
part = sys.argv[7]
participant = []
participant.append(part) 

included = [1,2,3,4]
included.remove(fold_shifted)

# main code ___________________________________________________________________
# First let's create mask of cortical vertices excluding medial wall

cortical_vertices = {}
for half in ['lh', 'rh']:
    test_ds = mv.niml.read('/idata/DBIC/cara/life/ridge/models/niml/ws.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(test_ds.samples[1:, :] != 0, axis=0) == 0] = 0

print('Model: {0}\nStim file: {1}, {2}\nHemi: {3}\nRuns in training: {4}\nRun in test: {5}\nParticipant: {6}'.format(model, stimfile1, stimfile2, hemi, included, fold_shifted, participant))

if model == 'visual':
    # stimfile1 = 'bg'
    X1train_stim, X1test_stim = get_visual_stim_for_fold('{0}_{1}'.format(model, stimfile1), fold_shifted, included)
    # stimfile2 = 'actions'
    X2train_stim, X2test_stim = get_visual_stim_for_fold('{0}_{1}'.format(model, stimfile2), fold_shifted, included)
    # train_stim, test_stim = get_visual_stim_for_fold('{0}_{1}'.format(model, stimfile), fold_shifted, included)
else:
    # stimfile1 = 'bg'
    X1train_stim, X1test_stim = get_narrative_stim_for_fold('{0}_{1}'.format(model, stimfile1), fold_shifted, included)
    # stimfile2 = 'actions'
    X2train_stim, X2test_stim = get_narrative_stim_for_fold('{0}_{1}'.format(model, stimfile2), fold_shifted, included)
    # train_stim, test_stim = get_narrative_stim_for_fold('{0}_{1}'.format(model, stimfile), fold_shifted, included)




# features ___________________________________________________________________
for test_p in participant:
    if align == 'ws':
        Ytrain_unconcat, Ytest = get_ws_data(test_p, fold_shifted, included, hemi)
        # train_resp, test_resp = get_ws_data(test_p, fold_shifted, included, hemi)
    elif align == 'aa':
        Ytrain_unconcat, Ytest = get_aa_data(test_p, fold_shifted, included, hemi)
        # train_resp, test_resp = get_aa_data(test_p, fold_shifted, included, hemi)
    else:
        print('\nLoading hyperaligned mappers...')
        mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(hemi, fold_shifted)))
        if align == 'ha_common':
            Ytrain_unconcat, Ytest = get_ha_common_data(test_p, mappers, fold_shifted, included, hemi)
        elif align == 'ha_testsubj':
            Ytrain_unconcat, Ytest = get_ha_common_data(test_p, mappers, fold_shifted, included, hemi)

        # train_resp, test_resp = get_aa_data(test_p, fold_shifted, included, hemi)

    # concatenate 3 runs
    X1train = np.concatenate(X1train_stim)
    X2train = np.concatenate(X2train_stim)
    Ytrain = np.concatenate(Ytrain_unconcat)
    print('\nShape of training and testing set')
    print(X1train.shape, "X1train" )
    print(X2train.shape, "X2train")
    print(X1test_stim.shape, "X1test_stim")
    print(X2test_stim.shape, "X2test_stim")
    print(Ytest.shape, "Ytest")
    print(Ytrain.shape, "Ytrain")

# tikreg
    alphas = np.logspace(0,3,20)
    ratios = np.logspace(-2,2,25)
    print('\nAlphas: np.logspace(0,3,20)')
    print('\nRatios: np.logspace(-2,2,25)')


    # angle = np.arctan(ratio)
    # lambda_one = np.cos(angle)*alpha
    # lambda_two = np.sin(angle)*alpha
    # bands = np.asarray([lambda_one]*X1train.shape[1] + [lambda_two]*X2train.shape[1])
    # Cinv = np.diag(bands**-1)
    #
    # A = np.hstack([X1train/lambda_one, X2train/lambda_two])
    # U, S, VT = np.linalg.svd(A, full_matrices=False)
    # V = VT.T
    # UTY = np.dot(U.T, Ytrain)
    # D = np.diag(S / (S**2 + alpha**2))
    #
    #         solution_svd_standard = np.linalg.multi_dot([V, D, UTY])*alpha
    #         solution_svd_bandstd2tik = np.dot(Cinv, solution_svd_standard)
    #
    #         print(np.rad2deg(angle), ratio, alpha, lambda_one, lambda_two)

            # Use tikreg to find the solution

    train_id = np.arange(X1train.shape[0])
    dur1, dur2, dur3 = tr_movie[included[0]]-3, tr_movie[included[1]]-3,tr_movie[included[2]]-3

    loro1 = [ (train_id[:dur1+dur2], train_id[dur1+dur2:]),                      
       ( np.concatenate((train_id[:dur1],train_id[dur1+dur2:]),axis=0),
         train_id[dur1:dur1+dur2]),                                           
        (train_id[dur1:], train_id[:dur1])] 

    X1_prior = spatial_priors.SphericalPrior(X1train, hyparams=ratios)
    X2_prior = spatial_priors.SphericalPrior(X2train, hyparams=ratios)
            # A temporal prior is unnecessary, so we specify no delays
    temporal_prior = temporal_priors.SphericalPrior(delays=[0]) # no delays

    fit_banded_polar = models.estimate_stem_wmvnp([X1train, X2train], Ytrain,
                                                          [X1test_stim, X2test_stim],Ytest,
                                                          feature_priors=[X1_prior, X2_prior],
                                                          temporal_prior=temporal_prior,
                                                          ridges=alphas,
                                                          folds=loro1,
                                                          performance=True,
                                                          weights=True,
                                                          verbosity=False)
    voxelwise_optimal_hyperparameters = fit_banded_polar['optima']
    print('\nVoxelwise optimal hyperparameter shape:', voxelwise_optimal_hyperparameters.shape)
    # kernel_weights = fit_banded_polar['weights']
    # primal_weights = []
    # for voxid, (temporal_lambda, lambda_one, lambda_two, alpha) in enumerate(voxelwise_optimal_hyperparameters):
    #     ws = np.dot(np.hstack([X1train/lambda_one**2, X2train/lambda_two**2]).T, kernel_weights[:,voxid]*alpha)
    #     primal_weights.append(ws)
    # primal_weights = np.asarray(primal_weights).T
    # print(primal_weights.shape)

    # Faster conversion of kernel weights to primal weights via matrix multiplication
    # each vector (new_alphas, lamda_ones, lamda_twos) contains v number of entries (e.g. voxels)
    new_alphas = voxelwise_optimal_hyperparameters[:,-1]
    lambda_ones = voxelwise_optimal_hyperparameters[:,1]
    lambda_twos = voxelwise_optimal_hyperparameters[:,2]

    # calculating primal weights from kernel weights
    kernel_weights = fit_banded_polar['weights']
    weights_x1 = np.linalg.multi_dot([X1train.T, kernel_weights, np.diag(new_alphas), np.diag(lambda_ones**-2)])
    weights_x2 = np.linalg.multi_dot([X2train.T, kernel_weights, np.diag(new_alphas), np.diag(lambda_twos**-2)])
    weights_joint = np.vstack([weights_x1, weights_x2])
    print("\nFeature1 weight shape: ",weights_x1.shape)
    print("\nFeature2 weight shape: ",weights_x2.shape)
    print("\nJoint weights shape: ", weights_joint.shape)
#    assert np.allclose(weights_joint, primal_weights)
    estimated_y1 = np.linalg.multi_dot([X1test_stim, weights_x1])
    estimated_y2 = np.linalg.multi_dot([X2test_stim, weights_x2])
    
    directory = os.path.join('/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/banded-ridge_alpha-cara_loro', '{0}/{1}/{2}_{3}/leftout_run_{4}'.format(align, model, stimfile1,stimfile2, fold_shifted), test_p, hemi)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # SAVE WEIGHTS
    print("shape: ",weights_x1.shape)
    print("type: ",type(weights_x1))
    np.save(os.path.join(directory, 'kernel-weights_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi_{5}.npy'.format(test_p, model, align, stimfile1, fold_shifted, hemi)), weights_x1)
    np.save(os.path.join(directory, 'kernel-weights_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi_{5}.npy'.format(test_p, model, align, stimfile2, fold_shifted, hemi)), weights_x2)
    pkl_file = open(os.path.join(directory, 'fit-dictionarity_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi_{5}.pkl'.format(test_p, model, align, stimfile1, fold_shifted, hemi)), 'rb')
    fit = pickle.load(pkl_file)
    pkl_file.close()

# correlation coefficiet 
    actual_df = pd.DataFrame(data = Ytest)
    estimated_y1_df = pd.DataFrame(data = estimated_y1)
    estimated_y2_df = pd.DataFrame(data = estimated_y2)
    corr_x1 = pd.DataFrame.corrwith(estimated_y1_df, actual_df, axis = 0, method = 'pearson')
    corr_x2 = pd.DataFrame.corrwith(estimated_y2_df, actual_df, axis = 0, method = 'pearson')

#    corr_x1 = np.corrcoef(estimated_y1, Ytest)
#    corr_x2 = np.corrcoef(estimated_y2, Ytest)

# save files

    med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
    out1 = np.zeros((corr_x1.shape[0]+med_wall_ind.shape[0]),dtype=np.dtype(corr_x1).type)
    out1[cortical_vertices[hemi] == 1] =corr_x1
    mv.niml.write(os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi_{5}.niml.dset'.format(test_p, model, align, stimfile1, fold_shifted, hemi)), out1[None,:])
    out = np.zeros((corr_x2.shape[0]+med_wall_ind.shape[0]),dtype=np.dtype(corr_x2).type)
    out[cortical_vertices[hemi] == 1] =corr_x2
    mv.niml.write(os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi_{5}.niml.dset'.format(test_p, model, align, stimfile2, fold_shifted, hemi)), out[None,:])

