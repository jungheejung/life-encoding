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
# data_dir = '/dartfs-hpc/scratch/cara/three_boots/'
data_dir = '/dartfs-hpc/scratch/cara/new_models'
# data_dir = '/idata/DBIC/cara/life/ridge/models/'
sam_data_dir = '/idata/DBIC/snastase/life'
n_vertices = 40962

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

# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_dat = mv.niml.read('/idata/DBIC/cara/life/ridge/models/new_niml/ws/ws_run1.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(test_dat.samples[1:, :] != 0, axis=0) == 0] = 0

# for hemi in hemispheres:
#     for modality in ['visual', 'narrative', 'visual_0vec']:
#         for model in ['actions', 'bg', 'all', 'actions_agents', 'actions_bg', 'agents_bg', 'agents']:
for hemi in hemispheres:
    for modality in ['visual_0vec']:
        for model in ['actions', 'bg', 'all', 'actions_agents', 'actions_bg', 'agents_bg', 'agents']:
            for run in [4]:
                print(hemi, modality, model)
                plist = []
                for participant in participants:
                    ds = np.load(os.path.join(data_dir, modality, model, 'run_{0}'.format(run), '{0}/{1}/corrs_w_nuisance.npy'.format(participant, hemi)))

                    if ds.shape[0] != 40962:
                        med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
                        dat = np.zeros((ds.shape[0]+med_wall_ind.shape[0]),dtype=ds.dtype)
                        dat[cortical_vertices[hemi] == 1] = ds
                        ds = dat
                    ds = np.nan_to_num(ds)
                    print(ds.shape)
                    plist.append(ds)

                concat = np.vstack(plist)
                print(concat.shape)
                avg = np.mean(concat, axis=0)[None,:]
                print(avg.shape)
                total = np.concatenate((avg, concat), axis=0)
                print(total.shape)

                np.save(os.path.join(data_dir, modality, 'npy', 'full_{0}_{1}_{2}.{3}.npy'.format(modality, model, run, hemi)), total)
                mv.niml.write(os.path.join(data_dir, modality, 'niml', 'full_{0}_{1}_{2}.{3}.niml.dset'.format(modality, model, run, hemi)), total)
