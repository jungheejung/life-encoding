# python life.py | tee logs/other_$(date +"%F-T%H%M%S").log
import matplotlib; matplotlib.use('agg')

import mvpa2.suite as mv
import pandas as pd
import numpy as np
from scipy import stats
from utils import hyper_ridge
import sys, os, time, csv

mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
sam_data_dir = '/idata/DBIC/snastase/life'
ridge_dir = '/idata/DBIC/cara/life/ridge'
cara_data_dir = '/idata/DBIC/cara/life/data/'


participants = sorted(['sub-rid000037', 'sub-rid000001', 'sub-rid000033', 'sub-rid000024',
                'sub-rid000041', 'sub-rid000032', 'sub-rid000006',
                'sub-rid000009', 'sub-rid000017', 'sub-rid000005', 'sub-rid000038',
                'sub-rid000031', 'sub-rid000012', 'sub-rid000027', 'sub-rid000014',
                'sub-rid000034', 'sub-rid000036', 'sub-rid000019'])

hemispheres = ['lh', 'rh']


tr = {1:374, 2:346, 3:377, 4:412}
n_samples = 1509
n_vertices = 40962
n_proc = 32     # how many cores do we have?
n_medial = {'lh': 3486, 'rh': 3491}

def load_data(filename):
    ds = mv.gifti_dataset(filename)
    ds.sa.pop('intents')
    ds.sa['subjects'] = [participant] * ds.shape[0]
    ds.fa['node_indices'] = range(n_vertices)
    # z-score features across samples
    mv.zscore(ds, chunks_attr=None)

    return ds

# def downsample(data, band):
#     max_time = data['onset'][-1] + data['duration'][-1]
#     tr_list = []
#     for i in np.arange(0, max_time, band):
#         dat = data[data['onset'] >= i and data['onset'] <= i+band]
#         filter_col = [col for col in s if col not in ['onset', 'duration']]
#         np_dat = np.array(dat[filter_col])
#         avg_dat = np.mean(np_dat, axis=0)
#         tr_list.append(avg_dat)
#     return(np.vstack(tr_list))
#
# def delayed(data):
#     return(np.concatenate((data[3:,:], data[2:-1,:], data[1:-2,:], data[:-3,:]), axis=1))

# SEMANTIC FEATURE VECTORS
for stimfile in os.listdir('/idata/DBIC/cara/life/data/semantic_cat/'):
# '/ihome/cara/life/w2v_src/google_w2v_ca.npy'
    cam = np.load(os.path.join('/idata/DBIC/cara/life/data/semantic_cat/', stimfile))
    cam_list = []
    cam_list.append(cam[:366,:])
    cam_list.append(cam[366:704,:])
    cam_list.append(cam[704:1073,:])
    cam_list.append(cam[1073:,:])

    for i in range(len(cam_list)):
    	this = cam_list[i]
    	cam_list[i] = np.concatenate((this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)

    train_stim = np.concatenate((cam_list[0], cam_list[1], cam_list[3]), axis=0)
    test_stim = cam_list[2]

    print(train_stim.shape, test_stim.shape)

    for hemi in hemispheres:
        # print('\nLoading hyperaligned mappers...')
        mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}.hdf5'.format(hemi)))
        all_corrs = []
        for test_p in participants:
            train_p = [x for x in participants if x != test_p]
            print('\nLoading fMRI GIFTI data and using {0} as test participant...'.format(test_p))
            train_resp = []
            for run in [1,2,4]:
                avg = []
                for participant in train_p:
                    if run == 4:
                        resp = mappers[participant].forward(load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-346vol_run-02.{1}.tproject.gii'.format(participant, hemi))).samples[4:-12,:])
                    else:
                        resp = mappers[participant].forward(load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[run], run, hemi))).samples[4:-7,:])
                    mv.zscore(resp, chunks_attr=None)
                    avg.append(resp)

                avg = np.mean(avg, axis=0)
                print(run, avg.shape)

                train_resp.append(avg)

            train_resp = np.concatenate(train_resp, axis=0)

            test_resp = mappers[test_p].forward(load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr[3], 3, hemi))).samples[4:-7,:])
            mv.zscore(test_resp, chunks_attr=None)
            print(test_resp.shape)
            run = 2
            balphas = np.logspace(-3, 3, 100)
            nboots = 15
            chunklen = 15
            nchunks = 25

            # print('\n\nRidge regression with alphas: {0}, nboots: {1}\n'.format(balphas, nboots))
            wt, corrs, alphas, bootstrap_corrs, valinds = hyper_ridge.bootstrap_ridge(train_stim, train_resp, test_stim, test_resp, balphas, nboots, chunklen, nchunks)

            # rr_model = Ridge(alpha=1.0)
            # rr_model.fit(w2v, fmri[:,i])
            print('\nFinished training ridge regression')
            print('\n\nwt: {0}'.format(wt))
            print('\n\ncorrs: {0}'.format(corrs))
            print('\n\nalphas: {0}'.format(alphas))
            # print('\n\nbootstrap_corrs: {0}'.format(bootstrap_corrs))
            # print('\n\nvalinds: {0}'.format(valinds))

            print('\n\nWriting to file...')
            directory = os.path.join(ridge_dir, 'avg-{}-slh-CV'.format(stimfile[:-4]), test_p, hemi)
            if not os.path.exists(directory):
                os.makedirs(directory)

            np.save(os.path.join(directory, 'weights.npy'), wt)
            np.save(os.path.join(directory, 'corrs.npy'), corrs)
            all_corrs.append(corrs)
            mv.niml.write(os.path.join(directory, 'corrs.{0}.niml.dset'.format(hemi)), corrs[None,:])

            np.save(os.path.join(directory, 'alphas.npy'), alphas)
            np.save(os.path.join(directory, 'bootstrap_corrs.npy'), bootstrap_corrs)
            np.save(os.path.join(directory, 'valinds.npy'), valinds)

            print('\nFinished writing to {0}'.format(directory))
        m = np.mean(np.vstack(all_corrs))
        np.save(os.path.join(ridge_dir, 'avg-{}-slh-CV'.format(stimfile[:-4]), 'avg_corrs.{0}.npy'.format(hemi)), m)

print('all done!')
