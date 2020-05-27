import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
import mvpa2.support.nibabel as mvsurf
import scipy.stats as st
from statsmodels.stats.multitest import multipletests
import os, random, sys

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

def return_avg(t):
    ds1 = np.load(os.path.join(data_dir, '{0}-leftout1'.format(t), p, h, 'corrs.npy'))
    ds2 = np.load(os.path.join(data_dir, '{0}-leftout2'.format(t), p, h, 'corrs.npy'))
    ds3 = np.load(os.path.join(data_dir, '{0}-leftout3'.format(t), p, h, 'corrs.npy'))
    ds4 = np.load(os.path.join(data_dir, '{0}-leftout4'.format(t), p, h, 'corrs.npy'))
    tot = np.vstack((ds1, ds2, ds3, ds4))
    ds = np.mean(tot, axis=0)

    ds = np.nan_to_num(ds)
    return ds

for h in hemispheres:
    for comp in [['ws', 'aa'], ['ws', 'ha_common'], ['aa', 'ha_common'], ['aa', 'ha_testsubj'], ['ws', 'ha_testsubj'], ['ha_testsubj', 'ha_common']]:
        plist = []
        for p in participants:
            a = return_avg(comp[0])
            b = return_avg(comp[1])
            diff = np.subtract(b, a)
            print(comp, len([node for node in diff if node > 0.01]))

            plist.append(diff)

        concat = np.vstack(plist)
        avg = np.mean(concat, axis=0)[None,:]
        total = np.concatenate((avg, concat), axis=0)

        # np.save(os.path.join('~/diff/diff-{0}-{1}.{2}.npy'.format(comp[0], comp[1], h)), total)
        mv.niml.write(os.path.join('diff-{0}-{1}.{2}.niml.dset'.format(comp[0], comp[1], h)), total)
