#!/usr/bin/env python

from os.path import join
import numpy as np
import mvpa2.suite as mv
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from itertools import combinations

participants = {'1': 'sid000021', '2': 'sid000120',
                '3': 'sid000005', '4': 'sid000029',
                '5': 'sid000010', '6': 'sid000013',
                '7': 'sid000020', '8': 'sid000024',
                '9': 'sid000009', '10': 'sid000012',
                '11': 'sid000034', '12': 'sid000007'}

n_conditions = 90
n_vertices = 40962

base_dir = '/home/nastase/social_actions'
scripts_dir = join(base_dir, 'scripts')
data_dir = join(base_dir, 'fmri', '1021_actions', 'derivatives')
suma_dir = join(data_dir, 'freesurfer', 'fsaverage6', 'SUMA')
mvpa_dir = join(data_dir, 'pymvpa')

condition_order = mv.h5load(join(scripts_dir, 'condition_order.hdf5'))
reorder = condition_order['reorder']
sparse_ordered_labels = condition_order['sparse_ordered_labels']

# Load in searchlight RDMs
sl_rdms = {}
for hemi in ['lh', 'rh']:
    sl_rdms[hemi] = {}
    for participant in participants.keys():
        sl_result = mv.h5load(join(mvpa_dir, 'no_roi_ids', 'search_RDMs_sq_zscore_p{0}_{1}.hdf5'.format(participant, hemi)))
        sl_sq = sl_result.samples.reshape(n_conditions, n_conditions,  n_vertices)
        sl_tri = []
        for sl in sl_sq.T:
            sl_tri.append(squareform(sl, checks=False))
        sl_tri = np.array(sl_tri).T
        assert sl_tri.shape == (n_conditions * (n_conditions - 1) / 2, n_vertices)
        sl_tri = mv.Dataset(sl_tri,
                            sa={'conditions': list(combinations(condition_order['original_condition_order'], 2))},
                            fa=sl_result.fa, a=sl_result.a)
        sl_tri.sa['participants'] = [int(participant)] * sl_tri.shape[0]
        sl_rdms[hemi][participant] = sl_tri
        print("Loaded searchlight RDMs for participant {0} "
              "hemisphere {1}".format(participant, hemi))

# Compute ISC in leave-one-out fashion
sl_iscs = {}
for hemi in ['lh', 'rh']:
    sl_iscs[hemi] = {}
    for participant in sorted(participants.keys()):
        lo_rdm = sl_rdms[hemi][participant].samples
        mean_rdm = np.mean(np.dstack([sl_rdms[hemi][p].samples
                                      for p in sorted(participants.keys())
                                      if p is not participant]), axis=2)
        assert lo_rdm.shape == mean_rdm.shape
        print lo_rdm.shape, mean_rdm.shape
        sl_isc = np.array([spearmanr(np.nan_to_num(lo),
                                     np.nan_to_num(mean))[0]
                           for lo, mean in zip(lo_rdm.T, mean_rdm.T)])[None, :]
        sl_iscs[hemi][participant] = sl_isc
        mv.niml.write(join(mvpa_dir, 'search_ISC_zscore_nohyper_{0}_p{1}.niml.dset'.format(hemi, participant)), sl_isc)
        print("Finished computing RDM ISC for "
              "participant {0} hemisphere {1}".format(participant, hemi))

for hemi in ['lh', 'rh']:
    mean_isc = np.mean(np.vstack([sl_iscs[hemi][p]
                                     for p in sorted(participants.keys())]),
                          axis=0)[None, :]
    mean_isc = np.nan_to_num(mean_isc)
    assert mean_isc.shape == (1, n_vertices)
    print("Maximum searchlight ISC = {0}".format(np.amax(mean_isc)))
    mv.niml.write(join(mvpa_dir, 'search_ISC_mean_zscore_nohyper_{0}.niml.dset'.format(hemi)), mean_isc)
