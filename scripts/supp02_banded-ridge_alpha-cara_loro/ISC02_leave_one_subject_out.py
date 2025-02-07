#!/usr/bin/env python
import numpy as np
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('agg')
import mvpa2.suite as mv
import sys
import os
import glob


# directories _________________________________________________________________________
sam_data_dir = '/idata/DBIC/snastase/life'
data_dir = '/idata/DBIC/cara/life/ridge/models'
mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
result_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/banded-ridge_alpha-cara_loro'

# parameter ___________________________________________________________________________
tr = {1: 374, 2: 346, 3: 377, 4: 412}

n_samples = 1509
n_vertices = 40962
n_medial = {'lh': 3486, 'rh': 3491}
participants = sorted(['sub-rid000037', 'sub-rid000001', 'sub-rid000033', 'sub-rid000024',
                'sub-rid000019', 'sub-rid000041', 'sub-rid000032', 'sub-rid000006',
                'sub-rid000009', 'sub-rid000017', 'sub-rid000005', 'sub-rid000038',
                'sub-rid000031', 'sub-rid000012', 'sub-rid000027', 'sub-rid000014',
                'sub-rid000034', 'sub-rid000036'])
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_ds = mv.niml.read('/idata/DBIC/cara/life/ridge/models/niml/ws.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(test_ds.samples[1:, :] != 0, axis=0) == 0] = 0


cortical_vertices = {}
for hemi in ['lh', 'rh']:
    # ws = mv.niml.read('/idata/DBIC/cara/life/ridge/models/new_niml/ws/ws_run1.{0}.niml.dset'.format(hemi))
    ws = mv.niml.read(os.path.join(
        data_dir, 'niml', 'ws.' + hemi + '.niml.dset'))
    cortical_vertices[hemi] = np.ones((40962))
    cortical_vertices[hemi][np.sum(ws.samples[1:, :] != 0, axis=0) == 0] = 0

models = ['agents','actions', 'bg']
aligns = [ 'aa', 'ha_testsubj']  # ['aa', 'ws', 'ha_testsubj', 'ha_common'] 'ha_testsubj',
runs = range(1, 5)
hemispheres = ['lh', 'rh']

# load data ___________________________________________________________________________

plist = []
for align in aligns:
    if not os.path.exists(os.path.join(result_dir, align, 'isc_0104')):
        os.makedirs(os.path.join(result_dir,  align,  'isc_0104'))
    for model in models:
        print(model)
        for hemi in hemispheres:
            runlist = []
            for run in runs:
                avg_stack = np.empty((4, 40962))
                triu_corrs = []
	            plist = []
                for p in participants:
                    filenames = []
                    filenames = glob.glob(os.path.join(result_dir, align, 'visual', 'bg_actions_agents', 'leftout_run_' + str(run), '*', hemi,
                'kernel-weights_' + '*' + '_model-visual_align-' + align + '_feature-' + model + '_foldshifted-' + str(run) + '_hemi_' + hemi + '.npy'))

                # 1. stack across participants per run ____________________________
                # input: results > ridge-models > ws > visual > all > leftout_run_1 > sub-sid00001 > lh
                # output: results > ridge-models > group_npy > ws_1.lh.py
                for ind, fname in enumerate(filenames):
                    ds = np.load(fname)
                    plist.append(ds)
                all_data = np.array(plist)
                # all_data.shape # (18, 1200, 37476)

                temp_n_vertices = n_vertices - n_medial[hemi]
                subject_ids = len(filenames)
                isc_result = np.zeros((len(filenames), ds.shape[1]))
                subject_ids = np.arange(len(filenames))

                # stack
                for subject in subject_ids:
                    # subject_ids+1):
                    # left_out_subject.shape (1200, 37476)
                    # other_subjects.shape (17, 1200, 37476)
                    # other_avg.shape (1200, 37476)
                    left_out_subject = all_data[subject, :, :]
                    # get N - 1 subjects
                    other_subjects = all_data[subject_ids != subject, :, :]
                    # average N - 1 subjects (shape: 1200, 37471,)
                    other_avg = np.mean(other_subjects, axis=0)
                    # other_subjects_concat = np.concatenate(other_subjects)
                    for voxel in np.arange(temp_n_vertices):
                        left_out_voxel = left_out_subject[:, voxel]
                        other_avg_voxel = other_avg[:, voxel]
                        isc = pearsonr(left_out_voxel, other_avg_voxel)[0]  # get r-value from pearsonr
                        isc_result[subject, voxel] = isc

                triu_corrs = np.tanh(np.mean(np.nan_to_num(isc_result), axis=0))
                np.save(os.path.join(result_dir,align, 'isc_0104', 'isc_model-{0}_run-{1}_hemi-{2}_vsmean.npy'.format(model, run, hemi)), triu_corrs)
                med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
                output = np.zeros((triu_corrs.shape[0] + med_wall_ind.shape[0]), dtype = triu_corrs.dtype)
                output[cortical_vertices[hemi] == 1] = triu_corrs
                mv.niml.write(os.path.join(result_dir,align, 'isc_0104','{0}_isc_run{1}_vsmean.{2}.niml.dset'.format(model, run, hemi)), output[None,:])

                # at the end of run loop - stack files _________________________________________
                avg_stack[run-1] = output
            mv.niml.write(os.path.join(result_dir,align, 'isc_0104', 'group_{0}_isc_vsmean.{1}.niml.dset'.format(model, hemi)), np.mean(avg_stack, axis=0)[None,:])