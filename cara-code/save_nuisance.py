import matplotlib; matplotlib.use('agg')
import numpy as np
import pandas as pd
import mvpa2.suite as mv
import mvpa2.support.nibabel as mvsurf
import scipy.stats as st
from statsmodels.stats.multitest import multipletests
import os, random, sys

mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
# data_dir = '/dartfs-hpc/scratch/cara/three_boots/'
npy_dir = '/dartfs-hpc/scratch/cara/w2v/w2v_features'
cara_data_dir = '/dartfs-hpc/scratch/cara/data/'

data_dir = '/dartfs-hpc/scratch/cara/new_models'
# data_dir = '/idata/DBIC/cara/life/ridge/models/'
sam_data_dir = '/idata/DBIC/snastase/life'
n_vertices = 40962
tr_fmri = {1:374, 2:346, 3:377, 4:412}
tr_movie = {1:369, 2:341, 3:372, 4:406}


participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']

hemispheres = ['lh', 'rh']

def get_visual_stim_for_fold(stimfile, fold_shifted, included):
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
        full_stim[i] = np.concatenate((m_avg, this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)
        print(i+1, full_stim[i].shape)

    train_stim = [full_stim[i] for i in np.subtract(included, 1)]
    test_stim = full_stim[fold_shifted-1]

    return train_stim, test_stim

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
            groupby = tr_avg.shape[0] / tr_movie[run]
            remainder = tr_avg.shape[0] % tr_movie[run]
            tr_reshaped = np.reshape(tr_avg[:-remainder], (tr_movie[run], groupby))
            avg = np.mean(tr_reshaped, axis=1)
            print(avg.shape)
            mel_list[run-1] = avg
    return mel_list

def get_narrative_stim_for_fold(stimfile, fold_shifted, included):
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

def load_data(filename):
    ds = mv.gifti_dataset(filename)
    ds.sa.pop('intents')
    ds.sa['subjects'] = [participant] * ds.shape[0]
    ds.fa['node_indices'] = range(n_vertices)
    # z-score features across samples
    mv.zscore(ds, chunks_attr=None)

    return ds

# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_dat = mv.niml.read('/idata/DBIC/cara/life/ridge/models/new_niml/ws/ws_run1.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(test_dat.samples[1:, :] != 0, axis=0) == 0] = 0

for hemi in hemispheres:
    for modality in ['visual','narrative']:
        for model in ['actions', 'bg', 'all', 'actions_agents', 'actions_bg', 'agents_bg', 'agents']:
            for run in [4]:
                print(hemi, modality, model)
                included = [1,2,3,4]
                included.remove(run)
                plist = []
                for participant in participants:
                    if modality == 'visual':
                        train_stim, test_stim = get_visual_stim_for_fold('{0}_{1}'.format(modality, model), run, included)
                    else:
                        train_stim, test_stim = get_narrative_stim_for_fold('{0}_{1}'.format(modality, model), run, included)

                    if run == 4:
                        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr_fmri[run], run, hemi))).samples[4:-5,cortical_vertices[hemi] == 1]
                    else:
                        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr_fmri[run], run, hemi))).samples[4:-4,cortical_vertices[hemi] == 1]

                    directory = os.path.join('/dartfs-hpc/scratch/cara/new_models', '{0}/{1}/run_{2}'.format(modality,model, run), participant, hemi)
                    wt = np.load(os.path.join(directory, 'weights.npy'))

                    wo_pred = np.dot(test_stim[:,0][:,None], wt[0,:][None,:])

                    # Find prediction correlations
                    wo_nnpred = np.nan_to_num(wo_pred)

                    wo_corrs = np.nan_to_num(np.array([np.corrcoef(test_resp[:,ii], wo_nnpred[:,ii].ravel())[0,1]
                                                        for ii in range(test_resp.shape[1])]))
                    # # save the corrs without nuisance regressor
                    # np.save(os.path.join(directory, 'corrs_nuisance_only.npy'), wo_corrs)
                    # med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
                    # out = np.zeros((wo_corrs.shape[0]+med_wall_ind.shape[0]),dtype= wo_corrs.dtype)
                    # out[cortical_vertices[hemi] == 1] = wo_corrs
                    # mv.niml.write(os.path.join(directory, 'corrs_nuisance_only.{0}.niml.dset'.format(hemi)), out[None,:])
                    if wo_corrs.shape[0] != 40962:
                        med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
                        dat = np.zeros((wo_corrs.shape[0]+med_wall_ind.shape[0]),dtype=wo_corrs.dtype)
                        dat[cortical_vertices[hemi] == 1] = wo_corrs
                        wo_corrs = dat

                    wo_corrs = np.nan_to_num(wo_corrs)
                    print(wo_corrs.shape)
                    plist.append(wo_corrs)

                concat = np.vstack(plist)
                print(concat.shape)
                avg = np.mean(concat, axis=0)[None,:]
                print(avg.shape)
                total = np.concatenate((avg, concat), axis=0)
                print(total.shape)

                np.save(os.path.join(data_dir, modality, 'npy', 'nuisance_{0}_{1}_{2}.{3}.npy'.format(modality, model, run, hemi)), total)
                mv.niml.write(os.path.join(data_dir, modality, 'niml', 'nuisance_{0}_{1}_{2}.{3}.niml.dset'.format(modality, model, run, hemi)), total)
