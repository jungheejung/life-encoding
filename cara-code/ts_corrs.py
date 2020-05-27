import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
import scipy.stats as st
import sklearn.metrics as sk

import os

data_dir = '/idata/DBIC/cara/life/ridge/'
sam_data_dir = '/idata/DBIC/snastase/life'
types = ['avg-ana-CV']

participants = ['sub-rid000012']

hemispheres = ['lh', 'rh']
n_vertices = 40962
n = 12


def load_data(filename):
    ds = mv.gifti_dataset(filename)
    ds.sa.pop('intents')
    ds.sa['subjects'] = [participant] * ds.shape[0]
    ds.fa['node_indices'] = range(n_vertices)
    # z-score features across samples
    mv.zscore(ds, chunks_attr=None)

    return ds

cam = np.load('/ihome/cara/life/w2v_src/google_w2v_ca.npy')[704:1073,:]
Pstim = np.concatenate((cam[3:,:], cam[2:-1,:], cam[1:-2,:], cam[:-3,:]), axis=1)
for ty in types:
    for participant in participants:
        p_corrs = []
        for hemi in hemispheres:
            Presp = load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-377vol_run-03.{1}.tproject.gii'.format(participant, hemi))).samples[4:-7,:]
            wt = np.load(os.path.join(data_dir, ty, participant, hemi, 'weights.npy'))

            pred = np.dot(Pstim, wt)
            print('type: {0}'.format(ty))
            print('participant: {0}'.format(participant))
            print('hemi: {0}'.format(hemi))
            print('pred: {0}'.format(wt.shape))


            nnpred = np.nan_to_num(pred)
            print(nnpred.shape)
            corrs = []
            for jj in range(n, n+120, 4):
                c = np.nan_to_num(np.array([np.corrcoef(Presp[jj-n:jj+n,ii], nnpred[jj-n:jj+n,ii].ravel())[0,1] for ii in range(Presp.shape[1])]))
                print(c)
                corrs.append(c)
            corrs = np.vstack(corrs)
            # corrs = np.nan_to_num(np.array([np.corrcoef(Presp[:,ii], nnpred[:,ii].ravel())[0,1] for ii in range(Presp.shape[1])]))
            print(corrs.shape)
            mv.niml.write('corrs.{0}.niml.dset'.format(hemi), corrs)

            # print(np.mean(corrs), np.amax(corrs), len([_ for _ in corrs if _ < .1]))
#
# for ty in types:
#     for participant in participants:
#         for hemi in hemispheres:
#             Presp = load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-377vol_run-03.{1}.tproject.gii'.format(participant, hemi))).samples[4:-7,:]
#             wt = np.load(os.path.join(data_dir, ty, participant, hemi, 'weights.npy'))
#             corrs = np.load(os.path.join(data_dir, ty, participant, hemi, 'corrs.npy'))
#             print(corrs.shape)
#             ttest = st.ttest_1samp(corrs, np.zeros(len(corrs)))
#             print(ttest.pvalue)
