#!/usr/bin/env python

import glob
import os
import sys
import mvpa2.suite as mv
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('agg')


# directories _________________________________________________________________________
sam_data_dir = '/idata/DBIC/snastase/life'
data_dir = '/idata/DBIC/cara/life/ridge/models'
mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'

result_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/banded-ridge_alpha-cara_loro'

# parameter ___________________________________________________________________________
tr = {1: 374, 2: 346, 3: 377, 4: 412}

n_samples = 1509
n_vertices = 40962
tr_movie = {1: 369, 2: 341, 3: 372, 4: 406}
tr_fmri = {1: 374, 2: 346, 3: 377, 4: 412}
tr_length = 2.5

n_proc = 32     # how many cores do we have?
n_medial = {'lh': 3486, 'rh': 3491}

participants = sorted(['sub-rid000037', 'sub-rid000001', 'sub-rid000033', 'sub-rid000024',
                       'sub-rid000019', 'sub-rid000041', 'sub-rid000032', 'sub-rid000006',
                       'sub-rid000009', 'sub-rid000017', 'sub-rid000005', 'sub-rid000038',
                       'sub-rid000031', 'sub-rid000012', 'sub-rid000027', 'sub-rid000014',
                       'sub-rid000034', 'sub-rid000036'])

# cortical_vertices = {}
for hemi in ['lh', 'rh']:
    # ws = mv.niml.read('/idata/DBIC/cara/life/ridge/models/new_niml/ws/ws_run1.{0}.niml.dset'.format(hemi))
    ws = mv.niml.read(os.path.join(
        data_dir, 'niml', 'ws.' + hemi + '.niml.dset'))
    cortical_vertices[hemi] = np.ones((40962))
    cortical_vertices[hemi][np.sum(ws.samples[1:, :] != 0, axis=0) == 0] = 0

models = ['actions', 'bg', 'agents']  # ['all']
aligns = ['aa']  # ['aa', 'ws', 'ha_testsubj', 'ha_common']
runs = range(1, 5)
hemispheres = ['lh', 'rh']

# functions from Cara Van Uden Ridge Regression  ____________________________________________________________________


def get_visual_stim_for_fold(stimfile, fold_shifted, included):
    cam = np.load(os.path.join(npy_dir, '{0}.npy'.format(stimfile)))

    full_stim = []
    full_stim.append(cam[:369, :])
    full_stim.append(cam[369:710, :])
    full_stim.append(cam[710:1082, :])
    full_stim.append(cam[1082:, :])

    for i in range(len(full_stim)):
        this = full_stim[i]
        full_stim[i] = np.concatenate(
            (this[3:, :], this[2:-1, :], this[1:-2, :], this[:-3, :]), axis=1)

    train_stim = [full_stim[i] for i in np.subtract(included, 1)]
    test_stim = full_stim[fold_shifted - 1]

    return train_stim, test_stim


def get_mel():
    mel_list = [[], [], [], []]
    directory = os.path.join(cara_data_dir, 'spectral', 'complete')
    for f in os.listdir(directory):
        if 'csv' in f:
            run = int(f[-5])
            s = pd.read_csv(os.path.join(directory, f))
            filter_col = [col for col in s if col.startswith('mel')]
            tr_s = np.array(s[filter_col])
            tr_avg = np.mean(tr_s, axis=1)

            groupby = tr_avg.shape[0] / tr_movie[run]
            remainder = tr_avg.shape[0] % tr_movie[run]
            tr_reshaped = np.reshape(
                tr_avg[:-remainder], (tr_movie[run], groupby))
            avg = np.mean(tr_reshaped, axis=1)
            mel_list[run - 1] = avg
    return mel_list


def get_ws_data(test_p, fold_shifted, included, hemi):
    print(
        '\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        if run == 4:
            resp = mv.gifti_dataset(os.path.join(
                sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[run], run, hemi))).samples[4:-5, :]
        else:
            resp = mv.gifti_dataset(os.path.join(
                sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[run], run, hemi))).samples[4:-4, :]

        resp = resp[:, cortical_vertices[hemi] == 1]
        mv.zscore(resp, chunks_attr=None)
        print('train', run, resp.shape)

        train_resp.append(resp)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, :]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, :]

    test_resp = test_resp[:, cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)
    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def get_aa_data(test_p, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print(
        '\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]

            resp = resp[:, cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, :]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, :]

    test_resp = test_resp[:, cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def get_ha_common_data(test_p, mappers, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print(
        '\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]

            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            resp = resp[:, cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, :]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, :]

    mv.zscore(test_resp, chunks_attr=None)
    test_resp = mappers[participant].forward(test_resp)
    test_resp = test_resp[:, cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def get_ha_testsubj_data(test_p, mappers, fold_shifted, included, hemi):
    train_p = [x for x in participants if x != test_p]
    print(
        '\nLoading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]

            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            mv.zscore(resp, chunks_attr=None)
            resp = mappers[test_p].reverse(resp)
            resp = resp[:, cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, cortical_vertices[hemi] == 1]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print proc_stdout

# load data ___________________________________________________________________________

# load Y data (prob don't need to split, right?)
# leave one subject out


# if align == 'ws':
#     Ytrain_unconcat, Ytest = get_ws_data(
#         test_p, fold_shifted, included, hemi)
# elif align == 'aa':
#     Ytrain_unconcat, Ytest = get_aa_data(
#         test_p, fold_shifted, included, hemi)
# else:
#     print('\nLoading hyperaligned mappers...')
#     mappers = mv.h5load(os.path.join(
#         mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(hemi, fold_shifted)))
#     if align == 'ha_common':
#         Ytrain_unconcat, Ytest = get_ha_common_data(
#             test_p, mappers, fold_shifted, included, hemi)
#     elif align == 'ha_testsubj':
#         Ytrain_unconcat, Ytest = get_ha_common_data(
#             test_p, mappers, fold_shifted, included, hemi)
#
# Ytrain = np.concatenate(Ytrain_unconcat)
#
# Ytest

# load gifti data
plist = []
for align in aligns:
    if not os.path.exists(os.path.join(result_dir, align, 'isc_raw')):
        os.makedirs(os.path.join(result_dir,  align,  'isc_raw'))
    for model in models:
        print(model)
        for h in hemispheres:
            runlist = []
            for run in runs:
                plist = []
                for participant in participants:
                    #     filenames = []
                    #     filenames = glob.glob(os.path.join(result_dir, align, 'visual', 'bg_actions_agents', 'leftout_run_' + str(run), '*', h,
                    # 'kernel-weights_' + '*' + '_model-visual_align-' + align + '_feature-' + model + '_foldshifted-' + str(run) + '_hemi_' + h + '.npy'))
                    if align == 'aa':
                        if run == 4:
                            resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                                participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
                        else:
                            resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                                participant, tr_fmri[run], run, hemi))).samples[4:-4, :]

                        resp = resp[:, cortical_vertices[hemi] == 1]
                        mv.zscore(resp, chunks_attr=None)
                        plist.append(resp)
                        # np.shape(plist) (18, 403, 37471)

                    # ds = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    # test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, :]
                    # plist.append(ds)
                # 1. stack across participants per run ____________________________
                # input: results > ridge-models > ws > visual > all > leftout_run_1 > sub-sid00001 > lh
                # output: results > ridge-models > group_npy > ws_1.lh.py
                # for ind, file in enumerate(filenames):
                #     ds = np.load(file)
                #     plist.append(ds)
                all_data = np.array(plist)
                # all_data.shape # (18, 1200, 37476)

                temp_n_vertices = n_vertices - n_medial[hemi]
                subject_ids = len(participants)  # 18
                isc_result = np.zeros(
                    (len(participants), resp.shape[1]))  # (18, 37471)
                subject_ids = np.arange(len(participants))
                for subject in subject_ids:  # subject_ids+1):
                    # left_out_subject.shape (1200, 37476)
                    # other_subjects.shape (17, 1200, 37476)
                    # other_avg.shape (1200, 37476)
                    left_out_subject = all_data[subject, :, :]
                    # get N - 1 subjects
                    other_subjects = all_data[subject_ids != subject, :, :]
                    # average N - 1 subjects (shape: 1200, 37471,)
                    other_avg = np.mean(other_subjects, axis=0)
                    #other_subjects_concat = np.concatenate(other_subjects)
                    for voxel in np.arange(temp_n_vertices):
                        left_out_voxel = left_out_subject[:, voxel]
                        other_avg_voxel = other_avg[:, voxel]
                        isc = pearsonr(left_out_voxel, other_avg_voxel)[0]  # get r-value from pearsonr
                        isc_result[subject, voxel] = isc

                        # isc_result.shape (18, 37476)
                triu_corrs = np.tanh(
                    np.mean(np.nan_to_num(isc_result), axis=0))
                med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
                output = np.zeros(
                    (triu_corrs.shape[0] + med_wall_ind.shape[0]))
                output[cortical_vertices[hemi] == 1] = triu_corrs

                mv.niml.write(os.path.join(result_dir, align, 'isc_raw',
                           '{0}_isc_run{1}_vsmean.{2}.niml.dset'.format(model, run, hemi)), output)

#  work on this
            # avg_stack = np.empty((4, 40962))
            #
            # for run in range(1,5):
            #     print(mv.niml.read(os.path.join(result_dir,align,'isc_raw', '{0}_isc_run{1}_vsmean.{2}.niml.dset'.format(model, run, hemi))).shape)
            #     avg_stack[run-1] = mv.niml.read(os.path.join(result_dir,align,'isc', '{0}_isc_run{1}_vsmean.{2}.niml.dset'.format(model, run, hemi)))
            #     # Save it with niml.write
            #     print(avg_stack.shape, np.mean(avg_stack, axis=0).shape)
            #     mv.niml.write(os.path.join(result_dir,align, 'isc_raw', 'group_{0}_isc_vsmean.{1}.niml.dset'.format(model, hemi)), np.mean(avg_stack, axis=0)[None,:])
