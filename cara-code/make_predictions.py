# python life.py | tee logs/other_$(date +"%F-T%H%M%S").log
import matplotlib; matplotlib.use('agg')

import mvpa2.suite as mv
import numpy as np
from scipy import stats
from utils import hyper_ridge
import sys, os, time, csv
from sklearn.linear_model import RidgeCV

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

def get_stim_for_fold(stimfile, fold_shifted, included, include_motion):
    cam = np.load(os.path.join(npy_dir, '{0}.npy'.format(stimfile)))
    motion = np.load('/ihome/cara/global_motion/motion_downsampled_complete.npy')

    motion_list = []
    motion_list.append(motion[:369])
    motion_list.append(motion[369:710])
    motion_list.append(motion[710:1082])
    motion_list.append(motion[1082:])

    full_stim = []
    full_stim.append(cam[:369,:])
    full_stim.append(cam[369:710,:])
    full_stim.append(cam[710:1082,:])
    full_stim.append(cam[1082:,:])

    for i in range(len(full_stim)):
        m = motion_list[i]
        m_avg = np.mean(np.vstack((m[3:], m[2:-1], m[1:-2], m[:-3])),axis=0)
        m_avg = np.reshape(m_avg,(-1,1))

        this = full_stim[i]
        if include_motion:
            full_stim[i] = np.concatenate((m_avg, this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)
        else:
            full_stim[i] = np.concatenate((this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)
        print(i+1, full_stim[i].shape)

    train_stim = [full_stim[i] for i in np.subtract(included, 1)]
    test_stim = full_stim[fold_shifted-1]

    return train_stim, test_stim

stimfile = 'all'
fold_shifted = 4
included = [1,2,3]
include_motion = True

train_stim, test_stim = get_stim_for_fold(stimfile, fold_shifted, included, include_motion)

# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_ds = mv.niml.read('/dartfs-hpc/scratch/cara/models/niml/ws/ws_run1_singlealpha.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(test_ds.samples[1:, :] != 0, axis=0) == 0] = 0

for hemi in hemispheres:
    for test_p in participants:
        dir = os.path.join('/dartfs-hpc/scratch/cara/new_models', '{0}/run_{1}'.format(stimfile, fold_shifted), test_p, hemi)
        wt = np.load(os.path.join(dir, 'weights.npy'))

        print(test_stim[:,0][:, None].shape, wt[0,:][None,:].shape)
        motion_pred = np.dot(test_stim[:,0][:,None], wt[0,:][None,:])
        nnpred = np.nan_to_num(motion_pred)
        print(nnpred.shape)

        # save the corrs without nuisance regressor
        np.save(os.path.join(dir, 'motion_pred.npy'), nnpred)
        med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
        out = np.zeros((nnpred.shape[1]+med_wall_ind.shape[0]),dtype= nnpred.dtype)
        out[cortical_vertices[hemi] == 1] = nnpred
        mv.niml.write(os.path.join(dir, 'motion_pred.{0}.niml.dset'.format(hemi)), out[None,:])

        print('\nFinished writing to {0}'.format(dir))
