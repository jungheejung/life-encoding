import matplotlib; matplotlib.use('agg')

import mvpa2.suite as mv
import numpy as np
from scipy import stats
from utils import hyper_ridge
import sys, os, time, csv

mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
sam_data_dir = '/idata/DBIC/snastase/life'
cara_data_dir = '/idata/DBIC/cara/life'

participants = sorted(['sub-rid000037', 'sub-rid000001', 'sub-rid000033', 'sub-rid000024',
                'sub-rid000019', 'sub-rid000041', 'sub-rid000032', 'sub-rid000006',
                'sub-rid000009', 'sub-rid000017', 'sub-rid000005', 'sub-rid000038',
                'sub-rid000031', 'sub-rid000012', 'sub-rid000027', 'sub-rid000014',
                'sub-rid000034', 'sub-rid000036'])

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
t= []
for hemi in hemispheres:
    mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}.hdf5'.format(hemi)))

    print('\nLoading fMRI GIFTI data...')
    l = []
    for participant in participants:
        p = []
        for run in range(1, 5):
            p.append(mappers[participant].forward(load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[run], run, hemi)))))
        l.append(p)
    t.append(l)
mv.h5save("hyperaligned.hdf5", t)
