import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
import os

# data_dir = '/dartfs-hpc/scratch/cara/new_models'
data_dir = '/dartfs-hpc/scratch/cara/models'
n_vertices = 40962

participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']
# models = ['all', 'actions', 'animals', 'bg', 'actions_animals', 'actions_bg', 'animals_bg']
models = ['all']
aligns = ['aa', 'ha_testsubj','ws']
runs = range(1,5)
hemispheres = ['lh', 'rh']

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
for half in ['lh', 'rh']:
    test_dat = mv.niml.read('/idata/DBIC/cara/life/ridge/models/new_niml/ws/ws_run1.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(test_dat.samples[1:, :] != 0, axis=0) == 0] = 0

for align in aligns:
    for model in models:
        print(model)
        for h in hemispheres:
            runlist = []
            for run in runs:
                plist = []
                for p in participants:

                    # ds = np.load(os.path.join(data_dir, '{0}/run_{1}/{2}/{3}/{4}.npy'.format(model, run, p, h, nuisance)))
                    ds = np.load(os.path.join(data_dir, '{0}/{1}/{2}/leftout_run_{3}/{4}/{5}/corrs.npy'.format(align, 'visual', model, run, p, h)))

                    if ds.shape[0] != 40962:
                        med_wall_ind = np.where(cortical_vertices[h] == 0)[0]
                        dat = np.zeros((ds.shape[0]+med_wall_ind.shape[0]),dtype=ds.dtype)
                        dat[cortical_vertices[h] == 1] = ds
                        ds = dat
                    ds = np.nan_to_num(ds)
                    plist.append(ds)

                concat = np.vstack(plist)
                avg = np.mean(concat, axis=0)[None,:]
                total = np.concatenate((avg, concat), axis=0)
                print(total.shape)

                np.save(os.path.join(data_dir, 'npy', '{0}_{1}.{2}.npy'.format(align,run, h)), total)
                mv.niml.write(os.path.join(data_dir, 'niml', '{0}_{1}.{2}.niml.dset'.format(align, run, h)), total)

            #
            #     plist.append(avg)
            #
            # concat = np.vstack(plist)
            # print(concat.shape)
            # avg = np.mean(concat, axis=0)[None,:]
            # print(avg.shape)
            # total = np.concatenate((avg, concat), axis=0)
            # print(total.shape)
            #
            # np.save(os.path.join(data_dir, 'npy', '{0}.{1}.npy'.format(align, h)), total)
            # mv.niml.write(os.path.join(data_dir, 'niml', '{0}.{1}.niml.dset'.format(align, h)), total)
