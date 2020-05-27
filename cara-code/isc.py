import matplotlib; matplotlib.use('agg')
import mvpa2.suite as mv
import numpy as np
import sys, os
from scipy.stats import pearsonr

def test_generate_correlation_map():
    x = np.random.rand(10, 10)
    y = np.random.rand(20, 10)


sam_data_dir = '/idata/DBIC/snastase/life'
data_dir = '/idata/DBIC/cara/life/ridge/models'
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


# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for hemi in ['lh', 'rh']:
    ws = mv.niml.read('/idata/DBIC/cara/life/ridge/models/new_niml/ws/ws_run1.{0}.niml.dset'.format(hemi))
    cortical_vertices[hemi] = np.ones((40962))
    cortical_vertices[hemi][np.sum(ws.samples[1:, :] != 0, axis=0) == 0] = 0


# switch = int(sys.argv[1])
# run = (switch % 4) + 1
# if switch < 4:
#     model ='aa'
# else:
#     model = 'ha'
# print(run, model)

for model in ['ha']:
    for run in [3]:
        # Load in surface data sets
        if model =='ha':
            lh_mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_lh_leftout_{0}.hdf5'.format(run)))
            rh_mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_rh_leftout_{0}.hdf5'.format(run)))
        fc = 0
        fmri = []
        for participant in participants:
            print('\nLoading fMRI GIFTI data for {0}'.format(participant))
            if model == 'ha':
                rh = rh_mappers[participant].forward(load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.rh.tproject.gii'.format(participant, tr[run], run))))
                lh = lh_mappers[participant].forward(load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.lh.tproject.gii'.format(participant, tr[run], run))))
            else:
                rh = load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.rh.tproject.gii'.format(participant, tr[run], run)))
                lh = load_data(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.lh.tproject.gii'.format(participant, tr[run], run)))
            rh = rh.samples
            lh = lh.samples
            mv.zscore(rh, chunks_attr=None)
            mv.zscore(lh, chunks_attr=None)

            ds = np.concatenate((rh, lh), axis=1)

            fc += 2
            print('file {0}/{1} loaded'.format(fc, len(participants) * 2))

            fmri.append(ds)

        print(len(fmri))

        print('Computing pairwise correlations...')
        n_nodes = fmri[0].shape[1]
        n_sub = len(fmri)
        print(n_nodes)
        # from scipy.spatial.distance import pdist, squareform
        # pdist(dss, 'correlation')

        ds = np.empty((n_nodes, n_sub))

        for k in range(n_nodes):
            for i in range(n_sub):
                mask = np.arange(n_sub)
                mask = np.delete(mask,i)
                other = [fmri[j][:,k] for j in mask]
                con = np.vstack(other)

                ds[k, i] = np.arctanh(pearsonr(fmri[i][:, k], np.mean(con,axis=0))[0])

        # ds = np.empty((n_nodes, n_sub, n_sub))
        # for k in range(n_nodes):
        #     print(k)
        #     for i in range(n_sub):
        #         for j in range(n_sub):
        #             ds[k, i, j] = np.arctanh(pearsonr(fmri[i][:, k], fmri[j][:, k])[0])

        print(ds.shape)
        #
        # triu_corrs = []
        # for k in range(n_nodes):
        #     nums = np.nan_to_num(ds[k][np.triu_indices(n_sub, k=1)])
        #     nums = nums[nums < 1e308]
        #     triu_corrs.append(np.mean(nums))
        # triu_corrs = np.tanh(np.nan_to_num(triu_corrs))
        triu_corrs = np.tanh(np.mean(np.nan_to_num(ds),axis=1))
        rh = triu_corrs[:40962][None, :]
        lh = triu_corrs[40962:][None, :]

        print(rh.shape)
        # # Check a bunch of things for strangeness
        # assert lh.shape == (1, 37476)
        # assert type(lh) == np.ndarray
        # assert lh.dtype == 'float64'
        #
        # # Check a bunch of things for strangeness
        # assert rh.shape == (1, 37471)
        # assert type(rh) == np.ndarray
        # assert rh.dtype == 'float64'

        # Save it with niml.write
        mv.niml.write(os.path.join(data_dir, '{0}_isc_run{1}_vsmean.lh.niml.dset'.format(model, run)), lh)
        mv.niml.write(os.path.join(data_dir, '{0}_isc_run{1}_vsmean.rh.niml.dset'.format(model, run)), rh)
        #
for model in ['ha', 'aa']:
    lh_avg_stack = np.empty((4, 40962))
    rh_avg_stack = np.empty((4, 40962))
    for run in range(1,5):
        print(mv.niml.read(os.path.join(data_dir, '{0}_isc_run{1}_vsmean.rh.niml.dset'.format(model, run))).shape)
        rh_avg_stack[run-1] = mv.niml.read(os.path.join(data_dir, '{0}_isc_run{1}_vsmean.rh.niml.dset'.format(model, run)))
        lh_avg_stack[run-1] = mv.niml.read(os.path.join(data_dir, '{0}_isc_run{1}_vsmean.lh.niml.dset'.format(model, run)))
    # Save it with niml.write
    print(lh_avg_stack.shape, np.mean(lh_avg_stack, axis=0).shape)
    mv.niml.write(os.path.join(data_dir, 'isc/{0}_isc_vsmean.lh.niml.dset'.format(model)), np.mean(lh_avg_stack, axis=0)[None,:])
    mv.niml.write(os.path.join(data_dir, 'isc/{0}_isc_vsmean.rh.niml.dset'.format(model)), np.mean(rh_avg_stack, axis=0)[None,:])
