#!/usr/bin/env python

# load packages ________________________________________________________________________________
import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
import os

# parameters ___________________________________________________________________
data_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/banded-ridge-models'
n_vertices = 40962
participants = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006', 'sub-rid000009', 'sub-rid000012',
                'sub-rid000014', 'sub-rid000017', 'sub-rid000019', 'sub-rid000024', 'sub-rid000027',
                'sub-rid000031', 'sub-rid000032', 'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
                'sub-rid000037', 'sub-rid000038', 'sub-rid000041']
# models = ['all', 'actions', 'animals', 'bg', 'actions_animals', 'actions_bg', 'animals_bg']
models = ['actions', 'bg']  # ['all']

aligns = ['ws']  # ['aa', 'ws', 'ha_testsubj', 'ha_common']
runs = range(1, 5)
hemispheres = ['lh', 'rh']


# functions ____________________________________________________________________
def load_data(filename):
    ds = mv.gifti_dataset(filename)
    ds.sa.pop('intents')
    ds.sa['subjects'] = [participant] * ds.shape[0]
    ds.fa['node_indices'] = range(n_vertices)
    # z-score features across samples
    mv.zscore(ds, chunks_attr=None)

    return ds


# 0. exclude medial wall __________________________________________________________
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_dat = mv.niml.read(
        '/idata/DBIC/cara/life/ridge/models/new_niml/ws/ws_run1.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(
        test_dat.samples[1:, :] != 0, axis=0) == 0] = 0

plist = []
for align in aligns:
    for model in models:
        print(model)
        for h in hemispheres:
            runlist = []
            for run in runs:
                plist = []
                for p in participants:
                # 1. stack across participants per run ____________________________
                # input: results > ridge-models > ws > visual > all > leftout_run_1 > sub-sid00001 > lh
                # output: results > ridge-models > group_npy > ws_1.lh.py
                    # ds = np.load(os.path.join(data_dir, '{0}/run_{1}/{2}/{3}/{4}.npy'.format(model, run, p, h, nuisance)))
                    ds = mv.niml.read(os.path.join(data_dir, align, 'visual', 'bg_actions', 'leftout_run_' + str(run), p, h,
                    'corrcoef_' + p + '_model-visual_align-' + align + '_feature-' + model + '_foldshifted-' + str(run) + '_hemi_' + h + '.niml.dset'))

                        # data_dir, '{0}/{1}/{2}/leftout_run_{3}/{4}/{5}/corrs.npy'.format(align, 'visual', 'bg_actions' model, run, p, h)))
                    #
                    # if ds.shape[0] != 40962:
                    #     med_wall_ind = np.where(cortical_vertices[h] == 0)[0]
                    #     dat = np.zeros(
                    #         (ds.shape[0] + med_wall_ind.shape[0]), dtype=ds.dtype)
                    #     dat[cortical_vertices[h] == 1] = ds
                    #     ds = dat
                    ds_nan = np.nan_to_num(ds.samples[0])
                    plist.append(ds_nan)

                # concat.shape:  (18, 40962)
                concat = np.vstack(plist)
                # avg.shape:      (1, 40962)
                avg = np.mean(concat, axis=0)[None, :]
                # total.shape:   (19, 40962)
                total = np.concatenate((avg, concat), axis=0)
                # print(total.shape)

                group_dir = os.path.join(data_dir, 'group_corr_total')
                if not os.path.exists(group_dir):
                    os.makedirs(group_dir)

                np.save(os.path.join(group_dir,'group-corr_{0}_{1}_model-{2}.{3}.npy'.format(align, run, model, h)), total)
                mv.niml.write(os.path.join(data_dir, 'group_corr_total', 'group-corr_{0}_{1}_model-{2}.{3}.niml.dset'.format(align, run, model, h)), total)
            # 2. stack across participants & run ____________________________
            # output: results > ridge-models > group_npy > total_ws.lh.npy
                runlist.append(avg)
            total_concat = np.vstack(runlist)
            print(total_concat.shape) # (4, 40962)
            total_avg = np.mean(total_concat, axis=0)[None,:]
            print(total_avg.shape) # (1, 40962)
            total = np.concatenate((total_avg, total_concat), axis=0)
            # print(total.shape)

            np.save(os.path.join(data_dir, 'group_corr_total', 'total_{0}_model-{2}.{1}.npy'.format(align, h, model)), total)
            mv.niml.write(os.path.join(data_dir, 'group_corr_total', 'total_{0}_model-{2}.{1}.niml.dset'.format(align, h, model)), total)

