import matplotlib; matplotlib.use('agg')
import mvpa2.suite as mv
import numpy as np
import sys, os
from scipy.stats import pearsonr

sam_data_dir = '/idata/DBIC/snastase/life'
mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'

tr = {1:374, 2:346, 3:377, 4:412}

n_samples = 1509
n_vertices = 40962
participants = sorted(['sub-rid000037', 'sub-rid000001', 'sub-rid000033', 'sub-rid000024',
                'sub-rid000019', 'sub-rid000041', 'sub-rid000032', 'sub-rid000006',
                'sub-rid000009', 'sub-rid000017', 'sub-rid000005', 'sub-rid000038',
                'sub-rid000031', 'sub-rid000012', 'sub-rid000027', 'sub-rid000014',
                'sub-rid000034', 'sub-rid000036'])

def load_data(filename):
    ds = mv.gifti_dataset(filename)
    ds.sa.pop('intents')
    ds.sa['subjects'] = [participant] * ds.shape[0]
    ds.fa['node_indices'] = range(n_vertices)
    # z-score features across samples
    mv.zscore(ds, chunks_attr=None)

    return ds

# Load in surface data sets
fc = 0
fmri = []
for participant in participants:
    print('\nLoading fMRI GIFTI data for {0}'.format(participant))
    ts = []
    for run in range(1, 5):
        rh = load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.rh.tproject.gii'.format(participant, tr[run], run)))
        lh = load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.lh.tproject.gii'.format(participant, tr[run], run)))
        rh = rh.samples
        lh = lh.samples
        ds = np.concatenate((rh, lh), axis=1)
        ts.append(ds)
        fc += 2
        print('file {0}/{1} loaded'.format(fc, len(participants) * 8))
    ts = np.concatenate(ts)
    print(ts.shape)
    fmri.append(ts)

print(len(fmri))

print('Computing pairwise correlations...')
n_nodes = fmri[0].shape[1]
n_sub = len(fmri)
print(n_nodes)
ds = np.empty((n_nodes, n_sub, n_sub))
# from scipy.spatial.distance import pdist, squareform
# pdist(dss, 'correlation')
for k in range(n_nodes):
    print(k)
    for i in range(n_sub):
        for j in range(n_sub):
            ds[k, i, j] = pearsonr(fmri[i][:, k], fmri[j][:, k])[0]

print(ds.shape)

triu_corrs = []
for i in range(n_nodes):
    nums = ds[i][np.triu_indices(n_sub)]
    triu_corrs.append(np.sum(nums) / len(nums))
triu_corrs = np.nan_to_num(triu_corrs)

rh = triu_corrs[:40962][None, :]
lh = triu_corrs[40962:][None, :]

print(rh.shape)
# Check a bunch of things for strangeness
assert lh.shape == (1, 40962)
assert type(lh) == np.ndarray
assert lh.dtype == 'float64'

# Check a bunch of things for strangeness
assert rh.shape == (1, 40962)
assert type(rh) == np.ndarray
assert rh.dtype == 'float64'

# Save it with niml.write
mv.niml.write(os.path.join('ana_isc.lh.niml.dset'), lh)
mv.niml.write(os.path.join('ana_isc.rh.niml.dset'), rh)

ana_lh = mv.niml.read('niml/ana_isc.lh.niml.dset').samples
ana_rh = mv.niml.read('niml/ana_isc.rh.niml.dset').samples

diff_lh = np.subtract(lh, ana_lh)
diff_rh = np.subtract(rh, ana_rh)

# Save it with niml.write
mv.niml.write(os.path.join('niml/adiff_isc.lh.niml.dset'), diff_lh)
mv.niml.write(os.path.join('niml/diff_isc.rh.niml.dset'), diff_rh)
