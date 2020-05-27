import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
import mvpa2.support.nibabel as mvsurf
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests
import os, random, sys, operator, csv

mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
data_dir = '/dartfs-hpc/scratch/cara/models'
sam_data_dir = '/idata/DBIC/snastase/life'


participants = sorted(['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041'])

tr = {1:374, 2:346, 3:377, 4:412}

hemispheres = {'lh':0, 'rh':1}

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
for hemi in ['lh', 'rh']:
    ws = mv.niml.read('ws_1.{0}.niml.dset'.format(hemi))
    cortical_vertices[hemi] = np.ones((40962))
    cortical_vertices[hemi][np.sum(ws.samples[1:, :] != 0, axis=0) == 0] = 0

t = ['ws', 'aa', 'ha_testsubj']

for run in range(1,5):
    union_ind = []
    intersect_ind = []
    for i, p in enumerate(participants):

        corrs_a_lh = mv.niml.read('{0}_{1}.lh.niml.dset'.format(t[0], run)).samples[1 + i, cortical_vertices['lh'] == 1][None,:]
        corrs_a_rh = mv.niml.read('{0}_{1}.lh.niml.dset'.format(t[0], run)).samples[1 + i, cortical_vertices['rh'] == 1][None,:]
        corrs_a = np.concatenate((corrs_a_lh, corrs_a_rh), axis=1)

        corrs_b_lh = mv.niml.read('{0}_{1}.lh.niml.dset'.format(t[1], run)).samples[1 + i, cortical_vertices['lh'] == 1][None,:]
        corrs_b_rh = mv.niml.read('{0}_{1}.lh.niml.dset'.format(t[1], run)).samples[1 + i, cortical_vertices['rh'] == 1][None,:]
        corrs_b = np.concatenate((corrs_b_lh, corrs_b_rh), axis=1)

        corrs_c_lh = mv.niml.read('{0}_{1}.lh.niml.dset'.format(t[2], run)).samples[1 + i, cortical_vertices['lh'] == 1][None,:]
        corrs_c_rh = mv.niml.read('{0}_{1}.lh.niml.dset'.format(t[2], run)).samples[1 + i, cortical_vertices['rh'] == 1][None,:]
        corrs_c = np.concatenate((corrs_c_lh, corrs_c_rh), axis=1)

        corrs_a = np.reshape(corrs_a, corrs_a.shape[1])
        corrs_b = np.reshape(corrs_b, corrs_b.shape[1])
        corrs_c = np.reshape(corrs_c, corrs_c.shape[1])

        ind_a = np.argsort(corrs_a)[-10000:]
        ind_b = np.argsort(corrs_b)[-10000:]
        ind_c = np.argsort(corrs_c)[-10000:]

        print(ind_a.shape, ind_b.shape, ind_c.shape)
        print(corrs_c[ind_c])

        punion_ind = np.union1d(ind_a, np.union1d(ind_b, ind_c))
        union_ind.append(punion_ind)

        pinter_ind = np.intersect1d(ind_a, np.intersect1d(ind_b, ind_c))
        intersect_ind.append(pinter_ind)

        print("inter: ", pinter_ind.shape, "union: ",punion_ind.shape)


    with open("union{0}.csv".format(run), "wb") as f:
        writer = csv.writer(f)
        writer.writerows(union_ind)

    with open("intersection{0}.csv".format(run), "wb") as f:
        writer = csv.writer(f)
        writer.writerows(intersect_ind)




nverts = np.empty(216,)
c =0

for t in [['ws', 'aa'], ['ha_testsubj', 'ws'], ['aa', 'ha_testsubj']]:
    print(t)
    corrs_a_sum = np.empty(72,)
    corrs_b_sum = np.empty(72,)
    count = 0

    run_diff = np.empty((4,18))
    for run in range(1,5):
        diff_per = []
        total = 0
        with open('union{0}.csv'.format(run), 'r') as fp:
            reader = csv.reader(fp)
            temp = list(reader)

        vert_ind = []
        for l in temp:
            vert_ind.append(np.array([int(i) for i in l]))
        for i, p in enumerate(participants):
            corrs_a_lh = mv.niml.read('{0}_{1}.lh.niml.dset'.format(t[0], run)).samples[1 + i, cortical_vertices['lh'] == 1][None,:]
            corrs_a_rh = mv.niml.read('{0}_{1}.lh.niml.dset'.format(t[0], run)).samples[1 + i, cortical_vertices['rh'] == 1][None,:]
            corrs_a = np.concatenate((corrs_a_lh, corrs_a_rh), axis=1)

            corrs_b_lh = mv.niml.read('{0}_{1}.lh.niml.dset'.format(t[1], run)).samples[1 + i, cortical_vertices['lh'] == 1][None,:]
            corrs_b_rh = mv.niml.read('{0}_{1}.lh.niml.dset'.format(t[1], run)).samples[1 + i, cortical_vertices['rh'] == 1][None,:]
            corrs_b = np.concatenate((corrs_b_lh, corrs_b_rh), axis=1)

            corrs_a = np.reshape(corrs_a, corrs_a.shape[1])
            corrs_b = np.reshape(corrs_b, corrs_b.shape[1])

            diff_per.append(np.mean(np.subtract(corrs_b[vert_ind[i]], corrs_a[vert_ind[i]])))
            corrs_a_sum[count] = np.mean(corrs_a[vert_ind[i]])

            # diff_per.append(float(len([p for p in vert_ind[i] if corrs_b[p] > corrs_a[p]]))/float(vert_ind[i].shape[0]))
            # corrs_a_sum[count] = float(len([p for p in vert_ind[i] if corrs_b[p] > corrs_a[p]]))/float(vert_ind[i].shape[0])
            nverts[c] = vert_ind[i].shape[0]
            count+=1
            c+=1
        run_diff[run-1] = diff_per

    print(np.mean(np.mean(corrs_a_sum, axis=0)))
    print(ss.ttest_1samp(np.mean(run_diff, axis=0), 0.0))
print(np.mean(np.mean(nverts, axis=0)), np.std(nverts))
