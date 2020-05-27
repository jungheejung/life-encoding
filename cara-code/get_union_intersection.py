import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
import mvpa2.support.nibabel as mvsurf
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests
import os, random, sys, operator, csv

data = {'ws':None, 'aa':None, 'ha_testsubj':None}

# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for hemi in ['lh', 'rh']:
    ws = mv.niml.read('ws.{0}.niml.dset'.format(hemi))
    cortical_vertices[hemi] = np.ones((40962))
    cortical_vertices[hemi][np.sum(ws.samples[1:, :] != 0, axis=0) == 0] = 0

for model in data.keys():
    corrs_lh = mv.niml.read('{0}.lh.niml.dset'.format(model)).samples[:, cortical_vertices['lh'] == 1][None,:]
    corrs_rh = mv.niml.read('{0}.lh.niml.dset'.format(model)).samples[:, cortical_vertices['rh'] == 1][None,:]
    corrs = np.concatenate((corrs_lh, corrs_rh), axis=2)
    data[model] = np.reshape(corrs, corrs.shape[1:])

for i in range(0,19):
    ind = {'ws':None, 'aa':None, 'ha_testsubj':None}

    for model in data.keys():
        ind[model] = np.argsort(data[model][i,:])[-10000:]
        
    union_ind = np.union1d(ind['ws'], np.union1d(ind['aa'], ind['ha_testsubj']))

    inter_ind = np.intersect1d(ind['ws'], np.intersect1d(ind['aa'], ind['ha_testsubj']))

    print(union_ind.shape)
    if i in [0, 5, 6]:
        for model in data.keys():
            sub = data[model][i, union_ind]
            print('for {0} and ws: [{1} {2}], max {3}'.format(i, np.percentile(sub, 5), np.percentile(sub, 95), np.max(sub)))

    # with open("union_{0}.csv".format(i), "wb") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(union_ind)
    #
    # with open("intersection_{0}.csv".format(i), "wb") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(intersect_ind)
