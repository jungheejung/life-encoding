import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
import scipy.stats as ss
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

data_dir = '/idata/DBIC/cara/life/ridge/avg-slh-CV'
sam_data_dir = '/idata/DBIC/snastase/life'
mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'

participants = sorted(['sub-rid000037', 'sub-rid000001', 'sub-rid000033', 'sub-rid000024',
                'sub-rid000019', 'sub-rid000041', 'sub-rid000032', 'sub-rid000006',
                'sub-rid000009', 'sub-rid000017', 'sub-rid000005', 'sub-rid000038',
                'sub-rid000031', 'sub-rid000012', 'sub-rid000027', 'sub-rid000014',
                'sub-rid000034', 'sub-rid000036'])

hemispheres = ['lh', 'rh']
n_medial = {'lh': 3486, 'rh': 3491}

n_vertices = 40962

plen = len(participants)
nnodes = 40962
rh_pred = np.empty((plen, 366, nnodes))
lh_pred = np.empty((plen, 366, nnodes))

rh_avg_pred = np.empty((366, nnodes))
lh_avg_pred = np.empty((366, nnodes))

rh_resp = np.empty((plen, 366, nnodes))
lh_resp = np.empty((plen, 366, nnodes))

rh_avg_resp = np.empty((366, nnodes))
lh_avg_resp = np.empty((366, nnodes))

rh_correlations = np.empty((plen+1, nnodes))
lh_correlations = np.empty((plen+1, nnodes))

rh_pvals = np.empty((plen+1, nnodes))
lh_pvals = np.empty((plen+1, nnodes))

pred_dict = {'lh': lh_pred, 'rh': rh_pred}
resp_dict = {'lh': lh_resp, 'rh': rh_resp}
avg_pred_dict = {'lh': lh_avg_pred, 'rh': rh_avg_pred}
avg_resp_dict = {'lh': lh_avg_resp, 'rh': rh_avg_resp}
corrs_dict = {'lh': lh_correlations, 'rh': rh_correlations}
pvals_dict = {'lh': lh_pvals, 'rh': rh_pvals}



def load_from_model(filename):
    stored = np.load(filename)[None,:]
    print(stored.shape)

    stored = np.nan_to_num(stored)

    return stored

def load_data(filename):
    ds = mv.gifti_dataset(filename)
    ds.sa.pop('intents')
    ds.sa['subjects'] = [participant] * ds.shape[0]
    ds.fa['node_indices'] = range(n_vertices)
    # z-score features across samples
    mv.zscore(ds, chunks_attr=None)

    return ds

lh_data = np.empty((25, nnodes))
rh_data = np.empty((25, nnodes))

# create random vec of 1 and -1
# apply to 18 maps



def exclude_medial(ds, hemi):
    # Exclude medial wall
    medial_wall = np.where(np.sum(ds == 0, axis=0))
    cortical_vertices = np.where(np.sum(ds == 0, axis=0))
    print(len(medial_wall), n_medial[hemi])
    assert len(medial_wall) == n_medial[hemi]
    assert len(medial_wall) + len(cortical_vertices) == nnodes
    return medial_wall, cortical_vertices

this = np.load(os.path.join('/idata/DBIC/cara/life/data/semantic_cat/all.npy'))[704:1073,:]
Pstim = np.concatenate((this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)

for hemi in ['rh']:
    mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}.hdf5'.format(hemi)))
    for i, participant in enumerate(participants):
        print(i, participant)

        wt = load_from_model(os.path.join(data_dir, '{0}/{1}/weights.npy'.format(participant, hemi)))
        Presp = mappers[participant].forward(load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-377vol_run-03.{1}.tproject.gii'.format(participant, hemi))).samples[4:-7,:])

        pred = np.dot(Pstim, wt)

        # Find prediction correlations
        nnpred = np.nan_to_num(pred)
        print('nnpred shape: ', nnpred.shape)

        rs = nnpred.reshape((nnpred.shape[0], -1), order='F')

        rrow = np.empty((1, nnodes))
        pval = np.empty((1, nnodes))

        for j in range(nnodes):
            rrow[:,j], pval[:,j] = ss.pearsonr(rs[:,j], Presp[:,j])

        pred_dict[hemi][i,:,:] = rs
        resp_dict[hemi][i,:,:] = Presp

        corrs_dict[hemi][i,:] = rrow
        pvals_dict[hemi][i,:] = pval

    avg_pred = np.mean((pred_dict[hemi]), axis=0)
    avg_resp = np.mean((resp_dict[hemi]), axis=0)

    avg_pred_dict[hemi] = avg_pred
    avg_resp_dict[hemi] = avg_resp

rrow = np.empty((1, nnodes))
pval = np.empty((1, nnodes))

for j in range(nnodes):
    rrow[:,j], pval[:,j] = ss.pearsonr(avg_pred[:,j], avg_resp[:,j])

    corrs_dict[hemi][18,:] = rrow
    pvals_dict[hemi][18,:] = pval
# permutation test
# correction for multiple
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
print(pvals.shape)
_, qvals, _, _ = multipletests(pvals, method='fdr_bh')

rh_qvals = qvals[:40962]
lh_qvals = qvals[40962:]

# ttest
# np.arctanh for Fisher's z transform
rh_ttest = st.ttest_1samp(np.arctanh(rh_correlations), 0.0)
lh_ttest = st.ttest_1samp(np.arctanh(lh_correlations), 0.0)

for i in range(plen+1):
    lh_data[i,:] = lh_correlations[i,:]
    rh_data[i,:] = rh_correlations[i,:]


lh_data[19,:] = lh_ttest.statistic
rh_data[19,:] = rh_ttest.statistic
lh_data[20,:] = lh_ttest.pvalue
rh_data[20,:] = rh_ttest.pvalue
lh_data[21,:] = lh_pvals[None,:]
rh_data[21,:] = rh_pvals[None,:]
lh_data[22,:] = st.norm.ppf(lh_pvals)
rh_data[22,:] = st.norm.ppf(rh_pvals)
lh_data[23,:] = lh_qvals[None,:]
rh_data[23,:] = rh_qvals[None,:]
lh_data[24,:] = st.norm.ppf(lh_qvals)
rh_data[24,:] = st.norm.ppf(rh_qvals)

lh_data = np.nan_to_num(lh_data)
rh_data = np.nan_to_num(rh_data)

mv.niml.write('slh_corr_analysis.lh.niml.dset', lh_data)
mv.niml.write('slh_corr_analysis.rh.niml.dset', rh_data)

mv.niml.write('analysis/correlations.lh.niml.dset', lh_correlations)
mv.niml.write('analysis/correlations.rh.niml.dset', rh_correlations)

mv.niml.write('analysis/pvals.lh.niml.dset', lh_pvals)
mv.niml.write('analysis/pvals.rh.niml.dset', rh_pvals)

pvals = np.concatenate((rh_pvals, lh_pvals), axis=1)
print(pvals.shape)

rh_qvals = np.empty((plen+1, nnodes))
lh_qvals = np.empty((plen+1, nnodes))

rh_z_qvals = np.empty((plen+1, nnodes))
lh_z_qvals = np.empty((plen+1, nnodes))

rh_z_pvals = np.empty((plen+1, nnodes))
lh_z_pvals = np.empty((plen+1, nnodes))

for i in range(plen+1):
    _, qvals, _, _ = multipletests(pvals[i,:], method='fdr_bh')

    rh_qvals[i,:] = qvals[:40962]
    lh_qvals[i,:] = qvals[40962:]

    rh_z_qvals[i,:] = ss.norm.ppf(qvals[:40962])
    lh_z_qvals[i,:] = ss.norm.ppf(qvals[40962:])

    rh_z_pvals[i,:] = ss.norm.ppf(rh_pvals[i,:])
    lh_z_pvals[i,:] = ss.norm.ppf(lh_pvals[i,:])

mv.niml.write('analysis/qvals.lh.niml.dset', lh_qvals)
mv.niml.write('analysis/qvals.rh.niml.dset', rh_qvals)

mv.niml.write('analysis/z_qvals.lh.niml.dset', lh_z_qvals)
mv.niml.write('analysis/z_qvals.rh.niml.dset', rh_z_qvals)

mv.niml.write('analysis/z_pvals.lh.niml.dset', lh_z_pvals)
mv.niml.write('analysis/z_pvals.rh.niml.dset', rh_z_pvals)

# load each word in word2vec (1, 300)
# concat with 1, 2, 3, 4 TR delays (1, 1200)
