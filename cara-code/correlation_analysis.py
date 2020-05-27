import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
import scipy.stats as st
from statsmodels.stats.multitest import multipletests
import os, random

# mean correlation
# t value (parametric) tt 1 sample
# ttest p value
# permutation test p val
# permutation test z val
# FDR perm for q values
# z values from q values 1.645 slide
# for each subject

data_dir = '/idata/DBIC/cara/life/ridge/no-slh'

participants = sorted(['sub-rid000037', 'sub-rid000001', 'sub-rid000033', 'sub-rid000024',
                'sub-rid000019', 'sub-rid000041', 'sub-rid000032', 'sub-rid000006',
                'sub-rid000009', 'sub-rid000017', 'sub-rid000005', 'sub-rid000038',
                'sub-rid000031', 'sub-rid000012', 'sub-rid000027', 'sub-rid000014',
                'sub-rid000034', 'sub-rid000036'])

hemispheres = ['lh', 'rh']
n_medial = {'lh': 3486, 'rh': 3491}

plen = len(participants)
nnodes = 40962


lh_data = np.empty((25, nnodes))
rh_data = np.empty((25, nnodes))

# create random vec of 1 and -1
# apply to 18 maps

rh_correlations = np.empty((plen, nnodes))
lh_correlations = np.empty((plen, nnodes))

def exclude_medial(ds, hemi):
    # Exclude medial wall
    medial_wall = np.where(np.sum(ds == 0, axis=0))
    cortical_vertices = np.where(np.sum(ds == 0, axis=0))
    print(len(medial_wall), n_medial[hemi])
    assert len(medial_wall) == n_medial[hemi]
    assert len(medial_wall) + len(cortical_vertices) == nnodes
    return medial_wall, cortical_vertices

for i, participant in enumerate(participants):
    print(i, participant)
    corrsfile = os.path.join(data_dir, '{0}/corrs.npy'.format(participant))

    corrs = np.load(corrsfile)
    print(corrs.shape)
    rh = corrs[:40962][None, :]
    lh = corrs[40962:][None, :]

    # rh_medial, rh_cortical = exclude_medial(rh, 'rh')
    # lh_medial, lh_cortical = exclude_medial(lh, 'lh')
    #
    # print(rh_medial.shape, lh_medial.shape)
    # Check a bunch of things for strangeness
    assert lh.shape == (1, 40962)
    assert type(lh) == np.ndarray
    assert lh.dtype == 'float64'
    lh = np.nan_to_num(lh)

    # Check a bunch of things for strangeness
    assert rh.shape == (1, 40962)
    assert type(rh) == np.ndarray
    assert rh.dtype == 'float64'
    rh = np.nan_to_num(rh)

    rh_correlations[i,:] = rh
    lh_correlations[i,:] = lh


# permutation test
# correction for multipl
lh_mean_corrs = np.empty((1000, nnodes))
rh_mean_corrs = np.empty((1000, nnodes))

for j in range(1000):

    rh_rand = np.empty((plen, nnodes))
    lh_rand = np.empty((plen, nnodes))

    for i in range(plen):
        rnd = np.random.choice([-1,1],size=nnodes)
        rh_rand[i,:] = np.multiply(rh_correlations[i,:], rnd)

        rnd = np.random.choice([-1,1],size=nnodes)
        lh_rand[i,:] = np.multiply(lh_correlations[i,:], rnd)

    lh_mean_corrs[j,:] = np.mean(lh_rand, axis=0)
    rh_mean_corrs[j,:] = np.mean(rh_rand, axis=0)

lh_emp_mean = np.mean(lh_correlations, axis=0)
rh_emp_mean = np.mean(rh_correlations, axis=0)

rh_bools = np.empty((1000, nnodes))
lh_bools = np.empty((1000, nnodes))

for j in range(1000):
    lh_bools[j,:] = np.absolute(lh_mean_corrs[j,:]) >= np.absolute(lh_emp_mean)
    rh_bools[j,:] = np.absolute(rh_mean_corrs[j,:]) >= np.absolute(rh_emp_mean)

lh_pvals = (np.sum(lh_bools, axis=0) + 1) / 1001
rh_pvals = (np.sum(rh_bools, axis=0) + 1) / 1001

pvals = np.concatenate((rh_pvals, lh_pvals))
# adjusted q values
_, qvals, _, _ = multipletests(pvals, method='fdr_bh')

rh_qvals = qvals[:40962]
lh_qvals = qvals[40962:]

# ttest
# np.arctanh for Fisher's z transform
rh_ttest = st.ttest_1samp(np.arctanh(rh_correlations), 0.0)
lh_ttest = st.ttest_1samp(np.arctanh(lh_correlations), 0.0)

for i in range(plen):
    lh_data[i,:] = lh_correlations[i,:]
    rh_data[i,:] = rh_correlations[i,:]
lh_data[18,:] = lh_emp_mean
rh_data[18,:] = rh_emp_mean
lh_data[19,:] = lh_ttest.statistic
rh_data[19,:] = rh_ttest.statistic
lh_data[20,:] = lh_ttest.pvalue
rh_data[20,:] = rh_ttest.pvalue
lh_data[21,:] = lh_pvals
rh_data[21,:] = rh_pvals
lh_data[22,:] = st.norm.ppf(lh_pvals)
rh_data[22,:] = st.norm.ppf(rh_pvals)
lh_data[23,:] = lh_qvals
rh_data[23,:] = rh_qvals
lh_data[24,:] = st.norm.ppf(lh_qvals)
rh_data[24,:] = st.norm.ppf(rh_qvals)

lh_data = np.nan_to_num(lh_data)
rh_data = np.nan_to_num(rh_data)

mv.niml.write('no-slh_corr_analysis.lh.niml.dset', lh_data)
mv.niml.write('no-slh_corr_analysis.rh.niml.dset', rh_data)

# load each word in word2vec (1, 300)
# concat with 1, 2, 3, 4 TR delays (1, 1200)
