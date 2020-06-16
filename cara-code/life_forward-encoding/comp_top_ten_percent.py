import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
import mvpa2.support.nibabel as mvsurf
import scipy.stats as st
from statsmodels.stats.multitest import multipletests
import os, random, sys

# mean correlation
# t value (parametric) tt 1 sample
# ttest p value
# permutation test p val
# permutation test z val
# FDR perm for q values
# z values from q values 1.645 slide
# for each subject

# slh_lh = np.load('/idata/DBIC/cara/life/ridge/avg_corrs/npy/avg-slh-CV.lh.npy')
# slh_rh = np.load('/idata/DBIC/cara/life/ridge/avg_corrs/npy/avg-slh-CV.rh.npy')
# ana_rh = np.load('/idata/DBIC/cara/life/ridge/avg_corrs/npy/avg-ana-CV.rh.npy')
# ana_lh = np.load('/idata/DBIC/cara/life/ridge/avg_corrs/npy/avg-ana-CV.lh.npy')
#
# diff_lh = np.subtract(slh_lh, ana_lh)
# diff_rh = np.subtract(slh_rh, ana_rh)
#
# mv.niml.write('/idata/DBIC/cara/life/ridge/avg_corrs/niml/slh_ana_diff.lh.niml.dset', diff_lh)
# mv.niml.write('/idata/DBIC/cara/life/ridge/avg_corrs/niml/slh_ana_diff.rh.niml.dset', diff_rh)

mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
data_dir = '/dartfs-hpc/scratch/cara/three_boots/'
sam_data_dir = '/idata/DBIC/snastase/life'


participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']

hemispheres = ['lh', 'rh']

def load_data(filename):
    ds = mv.gifti_dataset(filename)
    ds.sa.pop('intents')
    ds.sa['subjects'] = [participant] * ds.shape[0]
    ds.fa['node_indices'] = range(n_vertices)
    # z-score features across samples
    mv.zscore(ds, chunks_attr=None)

    return ds

def get_stim_for_test_fold(fold):
    cam = np.load('/idata/DBIC/cara/life/semantic_cat/all.npy')
    full_stim = []
    full_stim.append(cam[:366,:])
    full_stim.append(cam[366:704,:])
    full_stim.append(cam[704:1073,:])
    full_stim.append(cam[1073:,:])

    for i in range(len(full_stim)):
        this = full_stim[i]
        full_stim[i] = np.concatenate((this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)
        print(i+1, full_stim[i].shape)

    test_stim = full_stim[fold-1]

    return test_stim

for h in hemispheres:
    for run in range(1,5):
        mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(h,run)))
        for t in ['ws', 'aa']:
            print('on hemisphere {0}, run {1}, and model type {2}'.format(h, run, t))
            for p in participants:
                wt = np.load(os.path.join(data_dir, '{0}-leftout{1}/{2}/{3}/weights.npy'.format(t, run, p, h)))
                Pstim = get_stim_for_test_fold(run)
                Presp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(p, tr[run], run, h))).samples

                pred = np.dot(Pstim, wt)

                mv.zscore(pred, chunks_attr=None)
                forward_pred = mappers[p].forward(pred)
                mv.zscore(pred, chunks_attr=None)

                # Find prediction correlations
                nnpred = np.nan_to_num(forward_pred)
                corrs = np.nan_to_num(np.array([np.corrcoef(Presp[:,ii], nnpred[:,ii].ravel())[0,1]
                                                    for ii in range(Presp.shape[1])]))
                np.save(os.path.join(data_dir, '{0}-leftout{1}/{2}/{3}/forward_corrs.npy'.format(t, run, p, h)), corrs)
