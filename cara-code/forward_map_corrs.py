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
data_dir = '/dartfs-hpc/scratch/cara/models/singlealpha'
sam_data_dir = '/idata/DBIC/snastase/life'


participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']

tr = {1:374, 2:346, 3:377, 4:412}

hemispheres = ['lh', 'rh']
n_vertices = 40962
switch = int(sys.argv[1])
run = (switch % 4) + 1
if switch < 4:
    h = hemispheres[0]
else:
    h = hemispheres[1]

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

    test_stim = full_stim[fold-1]

    return test_stim

# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for half in ['lh', 'rh']:
    ws = mv.niml.read('/idata/DBIC/cara/life/ridge/models/new_niml/ws/ws_run1.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(ws.samples[1:, :] != 0, axis=0) == 0] = 0

# print("Loading hyperalignment mappers...")
mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(h,run)))
for t in ['ws', 'aa']:
    print('on hemisphere {0}, run {1}, and model type {2}'.format(h, run, t))
    for p in participants:
        wt_missing_medial = np.load(os.path.join(data_dir, '{0}-leftout{1}_singlealpha/{2}/{3}/weights.npy'.format(t, run, p, h)))

        # put medial wall back in
        med_wall_ind = np.where(cortical_vertices[h] == 0)[0]
        wt = np.zeros((wt_missing_medial.shape[0], wt_missing_medial.shape[1]+med_wall_ind.shape[0]),dtype=wt_missing_medial.dtype)
        wt[:, cortical_vertices[h] == 1] = wt_missing_medial

        Pstim = get_stim_for_test_fold(run)
        if run == 4:
            Presp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(p, tr[run], run, h))).samples[4:-14,:]
        else:
            Presp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(p, tr[run], run, h))).samples[4:-7,:]

        mv.zscore(Presp, chunks_attr=None)
        forward_resp = mappers[p].forward(Presp)
        mv.zscore(forward_resp, chunks_attr=None)

        print("Loaded stim and resp data. Doing prediction...")
        pred = np.dot(Pstim, wt)
        print(Pstim.shape, wt.shape, pred.shape)

        mv.zscore(pred, chunks_attr=None)
        forward_pred = mappers[p].forward(pred)
        mv.zscore(forward_pred, chunks_attr=None)
        print(forward_pred.shape, Presp.shape)

        # Find prediction correlations
        nnpred = np.nan_to_num(forward_pred)
        corrs = np.nan_to_num(np.array([np.corrcoef(forward_resp[:,ii], nnpred[:,ii].ravel())[0,1] for ii in range(forward_resp.shape[1])]))
        np.save(os.path.join(data_dir, '{0}-leftout{1}_singlealpha/{2}/{3}/forward_corrs.npy'.format(t, run, p, h)), corrs)
        mv.niml.write(os.path.join(data_dir, '{0}-leftout{1}_singlealpha/{2}/{3}/forward_corrs.{3}.niml.dset'.format(t, run, p, h)), corrs)
