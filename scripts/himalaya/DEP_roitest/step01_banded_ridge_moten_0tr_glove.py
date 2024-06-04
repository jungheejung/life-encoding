#!/usr/bin/env python3


import os, sys, shutil, time
import argparse
import subprocess
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import PredefinedSplit
from himalaya.ridge import GroupRidgeCV
from himalaya.scoring import correlation_score, r2_score
import nibabel as nib
import matplotlib.pyplot as plt
import json
from os.path import join
import pathlib

start = time.time()


# Assign/create directories
user_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f000fb6'
scratch_dir = os.path.join('/dartfs-hpc/scratch', user_dir)
data_dir = '/dartfs/rc/lab/D/DBIC/DBIC/life_data/'
current_dir = os.getcwd()
#main_dir = pathlib.Path(current_dir).parents[1]
# Create scratch directory if it doesn't exist
if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)

# Hyperalignment directory (/idata/DBIC/cara/life/pymvpa/)
hyper_dir = os.path.join(data_dir, 'hyperalign_mapper')

# Life fMRI data directory (/idata/DBIC/snastase/life)
fmri_dir = os.path.join(data_dir, 'life_dataset')

# Semantic model directory (/idata/DBIC/cara/w2v/w2v_features)
# model_dir = os.path.join(data_dir, 'w2v_feature')
model_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/annotations/glove'
# Set up some hard-coded experiment variables
subjects = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006',
            'sub-rid000009', 'sub-rid000012', 'sub-rid000014',
            'sub-rid000017', 'sub-rid000019', 'sub-rid000024',
            'sub-rid000027', 'sub-rid000031', 'sub-rid000032',
            'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
            'sub-rid000037', 'sub-rid000038', 'sub-rid000041']

tr = 2.5
model_durs = {1: 369, 2: 341, 3: 372, 4: 406}
# model_durs = {1: 369, 2: 341, 3: 372, 4: 381} 
fmri_durs = {1: 374, 2: 346, 3: 377, 4: 412} # 385 8< 
n_samples = np.sum(list(fmri_durs.values()))
n_vertices = 40962
n_medial = {'lh': 3486, 'rh': 3491}
n_blocks = 3 #20# 8 hr instead of 30 min ... break up n_vertices
model_ndim = 300
run_labels = [1, 2, 3, 4]
roi_json = os.path.join('/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/himalaya', 'rois_test.json')

# Parameters for optional PCA
run_pca = True
n_components = 40

# # Parameters from job submission script
# parser = argparse.ArgumentParser()
# parser.add_argument("--maindir", type=str,help="path to main directory of life encoding repo")
# parser.add_argument("-a", "--alignment",
#                     choices=['ws', 'aa', 'ha_test', 'ha_common'],
#                     help="within-subject, anatomical, or hyperalignment")
# parser.add_argument("--hemisphere", choices=['lh', 'rh'],
#                     help="specify hemisphere")
# parser.add_argument("-r", "--test-run", choices=[1, 2, 3, 4],
#                     type=int, help="specify test run")
# parser.add_argument("-s", "--test-subject", help="specify test subject")
# parser.add_argument("-f", "--features", nargs="*", type=str,
#                     default=['bg', 'actions', 'agents'],
#                     help="specify one or more feature spaces")
# parser.add_argument("--roi", default=None, type=str,
#                     help="ROI label from rois.json")
# parser.add_argument("--lag", default=None, type=int,
#                     help="TR lags")
# args = parser.parse_args()
# main_dir = args.maindir
# alignment = args.alignment # 'ws', 'aa', 'ha_test', 'ha_common'
# hemisphere = args.hemisphere # 'lh' or 'rh'
# test_run = args.test_run # 1, 2, 3, or 4
# test_run_id = test_run - 1 # zero-indexed test run
# train_runs = [r for r in run_labels if r != test_run] 
# test_subject = args.test_subject # e.g. 'sub-rid000005'
# features = args.features # e.g. ['bg', 'actions', 'agents'] 
# roi = args.roi # e.g. 'vt', '0'
# lag_test = args.lag

# Created save dir based on alignment

###############################
main_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding'
alignment = 'ha_common'
n_components = 40
hemisphere = 'lh'
test_run = 1
test_run_id = 0
train_runs = [2,3,4]
test_subject = 'sub-rid000005'
model_dir = join(main_dir, 'data/annotations/glove')
features = ['bg', 'actions', 'agents', 'moten'] 
roi = 'VMV_one' # MT_one AIP_one FFC_one PHG_one V1_one VMV_one
save_dir = os.path.join(main_dir, 'results', 'himalaya', f"roitest_{roi}_glove", f'{alignment}_pca-{n_components}')
# Create save directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
###############################
# Create mask of cortical vertices excluding medial wall    
medial_mask = np.load(os.path.join(data_dir, 'niml',
                                   f'fsaverage6_medial_{hemisphere}.npy'))
assert np.sum(medial_mask) == n_medial[hemisphere]
cortical_vertices = ~medial_mask

# Load ROIs and ensure ROI vertices are within cortical mask
if roi:
    with open(roi_json, 'r') as f:
        rois = json.load(f)
    
    roi_vertices = np.array(rois[hemisphere][roi])
    roi_shape = roi_vertices.shape
    
    roi_coords = np.where(cortical_vertices[roi_vertices])[0]
    cortical_vertices = np.intersect1d(np.where(cortical_vertices)[0],
                                       roi_vertices) # np.array(rois[hemisphere][roi]))
# Print critical analysis parameters
print("1. Analysis parameters")
print(f"Test run: {test_run} (train runs: {train_runs})\n"
      f"Test subject: {test_subject}\n"
      f"Hemisphere: {hemisphere}")


# Helper functions to read/write in GIfTI files
def read_gifti(gifti_fn):
    gii = nib.load(gifti_fn)
    data = np.vstack([da.data[np.newaxis, :]
                      for da in gii.darrays]) 
    return data

def write_gifti(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    gda = nib.gifti.GiftiDataArray(data)
    gii.add_gifti_data_array(gda)
    nib.gifti.giftiio.write(gii, output_fn)

    
# Function to (optionally) perform PCA on the stimulus model
def model_pca(train_model, test_model, n_components):
    
    # Z-score train data and apply to test data
    scaler = StandardScaler()
    train_model = scaler.fit_transform(train_model)
    test_model = scaler.transform(test_model)

    pca = PCA(n_components=n_components, svd_solver='full')
    
    train_model = pca.fit_transform(train_model)
    test_model = pca.transform(test_model)

    return train_model, test_model


# Load and assemble stimulus model with delays
def load_model(model_f, train_runs, test_run, model_durs,
               run_pca=False, n_components=None):
    
    # Load model and split into runs
    print(f"\n2. Semantic model ({model_f.split('_')[1].split('.')[0]})")
    model = np.load(os.path.join(model_dir, model_f))
    model_splits = np.cumsum(list(model_durs.values()))[:-1]
    model_runs = np.split(model, model_splits, axis=0)

    assert [len(r) for r in model_runs] == list(model_durs.values())

    # Split model runs into training and test sets
    train_model = model_runs[:test_run - 1] + model_runs[test_run:]
    test_model = model_runs[test_run - 1]
    for m in train_model:
        assert not np.array_equal(m, test_model)

    # Optionally reduce dimensionality of model with PCA
    if run_pca:

        # Concatenate training model across runs
        train_splits = np.cumsum([len(r) for r in train_model])[:-1]
        train_model = np.concatenate(train_model, axis=0)
        print(f"train_model: {train_model.shape[1]}, test_model:{test_model.shape[1]}, model_ndim: {model_ndim}")
        assert train_model.shape[1] == test_model.shape[1] #== model_ndim NOTE: confirm that it is correct to just check the train/model shape without the model ndim

        train_model, test_model = model_pca(train_model,
                                            test_model,
                                            n_components)
        
        # Re-split training model into runs
        train_model = np.split(train_model, train_splits)

    # Horizontally stack lagged versions of the stimulus model
    lags = [1, 2, 3, 4]
    train_cats, train_durs = [], {}
    for r, train_run in zip(train_runs, train_model):
        # train_cat = np.concatenate((train_run[lags[-1] - 1:-lags[0]],
        #                             train_run[lags[-2] - 1:-lags[1]],
        #                             train_run[lags[-3] - 1:-lags[2]],
        #                             train_run[lags[-4] - 1:-lags[3]]), axis=1)
        # Z-score each training run
        train_run = zscore(train_run, axis=0)
        n_trs = train_run.shape[0]
        n_wide = train_run.shape[1]

        train_cat = np.full((n_trs + lags[-1], n_wide * len(lags)), np.nan)
        train_cat[lags[0]:n_trs + lags[0], n_wide * (lags[0] - 1):n_wide * lags[0]] = train_run
        train_cat[lags[1]:n_trs + lags[1], n_wide * (lags[1] - 1):n_wide * lags[1]] = train_run
        train_cat[lags[2]:n_trs + lags[2], n_wide * (lags[2] - 1):n_wide * lags[2]] = train_run
        train_cat[lags[3]:n_trs + lags[3], n_wide * (lags[3] - 1):n_wide * lags[3]] = train_run   

        # TODO: standard scaler

        train_cats.append(train_cat)
        train_durs[r] = train_cat.shape[0]
        print(f"Train model run {r} shape:"
              f"\n\toriginal {train_run.shape} -> lagged {train_cat.shape}")
    train_model = np.concatenate(train_cats, axis=0)
    scaler = StandardScaler()
    train_model = np.nan_to_num(scaler.fit_transform(train_model))

        
    # test_model = np.concatenate((test_model[lags[-1] - 1:-lags[0]],
    #                              test_model[lags[-2] - 1:-lags[1]],
    #                              test_model[lags[-3] - 1:-lags[2]],
    #                              test_model[lags[-4] - 1:-lags[3]]), axis=1)
    n_trs = test_model.shape[0]
    test_cat = np.full((n_trs + lags[-1], n_wide * len(lags)), np.nan)
    test_cat[lags[0]:n_trs + lags[0], n_wide * (lags[0] - 1):n_wide * lags[0]] = test_model
    test_cat[lags[1]:n_trs + lags[1], n_wide * (lags[1] - 1):n_wide * lags[1]] = test_model
    test_cat[lags[2]:n_trs + lags[2], n_wide * (lags[2] - 1):n_wide * lags[2]] = test_model
    test_cat[lags[3]:n_trs + lags[3], n_wide * (lags[3] - 1):n_wide * lags[3]] = test_model
    # Z-score each model run separately
    # test_model = zscore(test_model, axis=0)
    # test_model = zscore(test_cat, axis=0)
    test_model = np.nan_to_num(scaler.transform(test_cat))
    print("Concatenated training model shape:", train_model.shape,
          f"\nTest model run {test_run} shape:", test_model.shape)

    return train_model, train_durs, test_model


# Load fMRI data for only test subject (within-subject encoding model)
def load_ws_data(test_subject, test_run, train_runs,
                 hemisphere, cortical_vertices):

    # Load training runs for the test subject
    print("\n3. Within-subject fMRI data")
    print(f"Loading within-subject data for {test_subject}")
    train_data = []
    for train_run in train_runs:
        if train_run == 4:

            # Changing to non-PyMVPA read_gifti loading
            run_data = read_gifti(os.path.join(
                fmri_dir, (f'{test_subject}_task-life_acq-'
                           f'{fmri_durs[train_run]}vol_run-'
                           f'{train_run:02d}.{hemisphere}.'
                           'tproject.gii')))[3:-7, :]
        else:
            run_data = read_gifti(os.path.join(
                fmri_dir, (f'{test_subject}_task-life_acq-'
                           f'{fmri_durs[train_run]}vol_run-'
                           f'{train_run:02d}.{hemisphere}.'
                           'tproject.gii')))[3:-6, :]

        # Extract cortical vertices (or ROI) for training runs
        run_data = run_data[:, cortical_vertices]
            
        # Z-score each training run prior to concatenation
        run_data = zscore(run_data, axis=0)
        
        print(f"Training run {train_run} shape: {run_data.shape}")
        train_data.append(run_data)
        
    # Concatenate training runs
    train_data = np.vstack(train_data)

    # Load test run GIfTI data using non-PyMVPA
    if test_run == 4:
        test_data = read_gifti(os.path.join(
            fmri_dir, (f'{test_subject}_task-life_acq-'
                       f'{fmri_durs[test_run]}vol_run-'
                       f'{test_run:02d}.{hemisphere}.'
                       'tproject.gii')))[3:-7, :]
    else:
        test_data = read_gifti(os.path.join(
            fmri_dir, (f'{test_subject}_task-life_acq-'
                       f'{fmri_durs[test_run]}vol_run-'
                       f'{test_run:02d}.{hemisphere}.'
                       'tproject.gii')))[3:-6, :]

    # Extract cortical vertices (or ROI) for test run
    test_data = test_data[:, cortical_vertices]
    
    # Z-score the test run over time
    test_data = zscore(test_data, axis=0)
    print(f"Test run {test_run} shape: {test_data.shape}")

    return train_data, test_data


# Load fMRI data for anatomically aligned between-subject encoding model
def load_aa_data(test_subject, test_run, train_runs,
                 hemisphere, cortical_vertices):
    
    # Load training runs for N - 1 subjects (excluding test subject)
    print("\n3. Anatomically-aligned across-subject fMRI data")
    print(f"Loading anatomically-aligned data for test subject {test_subject}")
    train_subjects = [s for s in subjects if s != test_subject]

    train_data = []
    for train_run in train_runs:
        train_mean = []
        for train_subject in train_subjects:
            if train_run == 4:
                
                # Changing to non-PyMVPA read_gifti loading
                run_data = read_gifti(os.path.join(
                    fmri_dir, (f'{train_subject}_task-life_acq-'
                               f'{fmri_durs[train_run]}vol_run-'
                               f'{train_run:02d}.{hemisphere}.'
                               'tproject.gii')))[3:-7, :]
            
            else:
                run_data = read_gifti(os.path.join(
                    fmri_dir, (f'{train_subject}_task-life_acq-'
                               f'{fmri_durs[train_run]}vol_run-'
                               f'{train_run:02d}.{hemisphere}.'
                               'tproject.gii')))[3:-6, :]

            # Extract cortical vertices (or ROI) for training runs
            run_data = run_data[:, cortical_vertices]

            # Z-score each training run prior to averaging/concatenation
            run_data = zscore(run_data, axis=0)
            
            # Populating training subjects for averaging
            train_mean.append(run_data)
            print(f"Loaded training run {train_run} for training "
                  f"subject {train_subject}")

        # Average training subjects and z-score
        train_mean = np.mean(train_mean, axis=0)
        train_mean = zscore(train_mean, axis=0)
        print(f"Averaged training run {train_run} across "
              f"{len(train_subjects)} anatomically-aligned "
              "training subjects")

        print(f"Training run {train_run} shape: {run_data.shape}")
        train_data.append(train_mean)

    # Concatenate training runs
    train_data = np.vstack(train_data)

    # Load test subject run GIfTI data using non-PyMVPA
    if test_run == 4:
        test_data = read_gifti(os.path.join(
            fmri_dir, (f'{test_subject}_task-life_acq-'
                       f'{fmri_durs[test_run]}vol_run-'
                       f'{test_run:02d}.{hemisphere}.'
                       'tproject.gii')))[3:-7, :]
        
    else:
        test_data = read_gifti(os.path.join(
            fmri_dir, (f'{test_subject}_task-life_acq-'
                       f'{fmri_durs[test_run]}vol_run-'
                       f'{test_run:02d}.{hemisphere}.'
                       'tproject.gii')))[3:-6, :]

    # Extract cortical vertices (or ROI) for test run
    test_data = test_data[:, cortical_vertices]
    
    # Z-score the test run over time
    test_data = zscore(test_data, axis=0)
    print(f"Test run {test_run} shape: {test_data.shape}")

    return train_data, test_data


# Load hyperaligned fMRI data projected into each subject's space
def load_ha_test_data(test_subject, test_run, train_runs,
                      hemisphere, cortical_vertices):
    
    # Load training runs for N - 1 subjects (excluding test subject)
    print("\n3. Hyperaligned (test-space) across-subject fMRI data")
    print(f"Loading hyperaligned data for test subject {test_subject}")
    train_subjects = [s for s in subjects if s != test_subject]

    train_data = []
    for train_run in train_runs:
        train_mean = []
        for train_subject in train_subjects:
            
            # TODO: GET RID OF DEGENERATE FIRST DIMENSION!!!
            if train_run == 4:
                
                # Changing to non-PyMVPA read_gifti loading
                run_data = read_gifti(os.path.join(
                    fmri_dir, (f'{train_subject}_task-life_acq-'
                               f'{fmri_durs[train_run]}vol_run-'
                               f'{train_run:02d}_'
                               f'desc-HAtestR{test_run}'
                               f"{test_subject.split('-')[1]}."
                               f'{hemisphere}.gii')))[0, 3:-7, :]
            
            else:
                run_data = read_gifti(os.path.join(
                    fmri_dir, (f'{train_subject}_task-life_acq-'
                               f'{fmri_durs[train_run]}vol_run-'
                               f'{train_run:02d}_'
                               f'desc-HAtestR{test_run}'
                               f"{test_subject.split('-')[1]}."
                               f'{hemisphere}.gii')))[0, 3:-6, :]

            # Extract cortical vertices (or ROI) for training runs
            run_data = run_data[:, cortical_vertices]

            # Z-score each training run prior to averaging/concatenation
            run_data = zscore(run_data, axis=0)
            
            # Populating training subjects for averaging
            train_mean.append(run_data)
            print(f"Loaded training run {train_run} for training "
                  f"subject {train_subject}")

        # Average training subjects and z-score
        train_mean = np.mean(train_mean, axis=0)
        train_mean = zscore(train_mean, axis=0)
        print(f"Averaged training run {train_run} across "
              f"{len(train_subjects)} hyperaligned "
              "training subjects")

        print(f"Training run {train_run} shape: {run_data.shape}")
        train_data.append(train_mean)

    # Concatenate training runs
    train_data = np.vstack(train_data)

    # Load test subject run GIfTI data using non-PyMVPA
    if test_run == 4:
        test_data = read_gifti(os.path.join(
            fmri_dir, (f'{test_subject}_task-life_acq-'
                       f'{fmri_durs[test_run]}vol_run-'
                       f'{test_run:02d}_'
                       f'desc-HAtestR{test_run}'
                       f"{test_subject.split('-')[1]}."
                       f'{hemisphere}.gii')))[0, 3:-7, :]
        
    else:
        test_data = read_gifti(os.path.join(
            fmri_dir, (f'{test_subject}_task-life_acq-'
                       f'{fmri_durs[test_run]}vol_run-'
                       f'{test_run:02d}_'
                       f'desc-HAtestR{test_run}'
                       f"{test_subject.split('-')[1]}."
                       f'{hemisphere}.gii')))[0, 3:-6, :]

    # Extract cortical vertices (or ROI) for test run
    test_data = test_data[:, cortical_vertices]
    
    # Z-score the test run over time
    test_data = zscore(test_data, axis=0)
    print(f"Test run {test_run} shape: {test_data.shape}")

    return train_data, test_data


# Load hyperaligned fMRI data projected into each subject's space
def load_ha_common_data(test_subject, test_run, train_runs,
                        hemisphere, cortical_vertices):
    
    # Load training runs for N - 1 subjects (excluding test subject)
    print("\n3. Hyperaligned (test-space) across-subject fMRI data")
    print(f"Loading hyperaligned data for test subject {test_subject}")
    train_subjects = [s for s in subjects if s != test_subject]

    train_data = []
    for train_run in train_runs:
        train_mean = []
        for train_subject in train_subjects:
            
            # TODO: GET RID OF DEGENERATE FIRST DIMENSION!!!
            if train_run == 4:
                
                # Changing to non-PyMVPA read_gifti loading
                run_data = read_gifti(os.path.join(
                    fmri_dir, (f'{train_subject}_task-life_acq-'
                               f'{fmri_durs[train_run]}vol_run-'
                               f'{train_run:02d}_'
                               f'desc-HAcommon{test_run}.'
                               f'{hemisphere}.gii')))[0, :-2, :]

            
            else:
                run_data = read_gifti(os.path.join(
                    fmri_dir, (f'{train_subject}_task-life_acq-'
                               f'{fmri_durs[train_run]}vol_run-'
                               f'{train_run:02d}_'
                               f'desc-HAcommon{test_run}.'
                               f'{hemisphere}.gii')))[0, :-1, :]


            # Extract cortical vertices (or ROI) for training runs
            run_data = run_data[:, cortical_vertices]

            # Z-score each training run prior to averaging/concatenation
            run_data = zscore(run_data, axis=0)
            
            # Populating training subjects for averaging
            train_mean.append(run_data)
            print(f"Loaded training run {train_run} for training "
                  f"subject {train_subject}")

        # Average training subjects and z-score
        train_mean = np.mean(train_mean, axis=0)
        train_mean = zscore(train_mean, axis=0)
        print(f"Averaged training run {train_run} across "
              f"{len(train_subjects)} hyperaligned "
              "training subjects")

        print(f"Training run {train_run} shape: {run_data.shape}")
        train_data.append(train_mean)

    # Concatenate training runs
    train_data = np.vstack(train_data)

    # Load test subject run GIfTI data using non-PyMVPA
    if test_run == 4:
        test_data = read_gifti(os.path.join(
            fmri_dir, (f'{test_subject}_task-life_acq-'
                       f'{fmri_durs[test_run]}vol_run-'
                       f'{test_run:02d}_'
                       f'desc-HAcommon{test_run}.'
                       f'{hemisphere}.gii')))[0, :-2, :]

        
    else:
        test_data = read_gifti(os.path.join(
            fmri_dir, (f'{test_subject}_task-life_acq-'
                       f'{fmri_durs[test_run]}vol_run-'
                       f'{test_run:02d}_'
                       f'desc-HAcommon{test_run}.'
                       f'{hemisphere}.gii')))[0, :-1, :]


    # Extract cortical vertices (or ROI) for test run
    test_data = test_data[:, cortical_vertices]
    
    # Z-score the test run over time
    test_data = zscore(test_data, axis=0)
    print(f"Test run {test_run} shape: {test_data.shape}")

    return train_data, test_data


# Load requested model features with one or more bands
train_bands, test_bands = [], []
for f in features:
    model_f = f'visual_{f}.npy'
    train_model, train_durs, test_model = load_model(model_f, train_runs,
                                                     test_run, model_durs,
                                                     run_pca=run_pca,
                                                     n_components=n_components)
    train_bands.append(train_model)
    test_bands.append(test_model)


# Load fMRI data with requested alignment method
if alignment == 'ws':
    train_data, test_data = load_ws_data(test_subject, test_run,
                                         train_runs, hemisphere,
                                         cortical_vertices)

if alignment == 'aa':
    train_data, test_data = load_aa_data(test_subject, test_run,
                                         train_runs, hemisphere,
                                         cortical_vertices)
    
if alignment == 'ha_test':
    train_data, test_data = load_ha_test_data(test_subject, test_run,
                                              train_runs, hemisphere,
                                              cortical_vertices)

if alignment == 'ha_common':
    train_data, test_data = load_ha_common_data(test_subject, test_run,
                                                train_runs, hemisphere,
                                                cortical_vertices)


# Function to reinsert cortical vertices into full ROI array for stacking  
def reinsert_vertices(roi_subset, roi_shape, roi_coords):    
    roi_result = np.full((roi_subset.shape[0], roi_shape[0]), np.nan)
    roi_result[:, roi_coords] = roi_subset
    return roi_result


# Run banded ridge regression
print("\n4. Banded ridge regression")

# Set up leave-one-run-out inner cross-validation loop
run_folds = np.concatenate([[r] * t for r, t in train_durs.items()])
assert len(run_folds) == train_bands[0].shape[0]
loro = PredefinedSplit(run_folds)

# Parameters for banded ridge model
n_iter = 100
print(f"Fitting banded ridge with {n_iter} iterations")
print(f"Feature space(s): {', '.join(features)}")

# Define banded ridge model and fit
ridge = GroupRidgeCV(groups="input", cv=loro, fit_intercept=False,
                     solver_params=dict(score_func=correlation_score,
                                        progress_bar=True, n_iter=n_iter))

# Fit the banded ridge regression model
ridge.fit(train_bands, train_data)
ridge_coef = ridge.coef_
print("Finished fitting banded ridge model")
print(f"Weight matrix shape: {ridge.coef_.shape}")

if roi:
    ridge_coef = reinsert_vertices(ridge_coef, roi_shape, roi_coords)

np.save(f'{save_dir}/ridge-coef_pca-{n_components}_align-{alignment}_{test_subject}_'
#np.save(f'{save_dir}/ridge-coef_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        ridge_coef)

# Use the fitted ridge model to predict left-out test run
print("Predicting left-out test run")
test_comb = ridge.predict(test_bands)
test_split = ridge.predict(test_bands, split=True)
print(f"Predicted test run {test_run} shape: {test_comb.shape}")
# Duplicate model predictions for for saving
test_comb_save = test_comb.copy()
test_split_save = {f: t for f, t in zip(features, test_split)}

if roi:
    test_comb_save = reinsert_vertices(test_comb, roi_shape, roi_coords)
    test_split_save = {f: reinsert_vertices(s, roi_shape, roi_coords)
                       for f, s in test_split_save.items()}
np.save(f'{save_dir}/comb-pred_pca-{n_components}_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        test_comb_save)
np.save(f'{save_dir}/split-pred_pca-{n_components}_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        test_split_save)

# Compute correlation-based performance and save
comb_r = correlation_score(test_data, test_comb)[np.newaxis, :]
split_r = {f: correlation_score(test_data, ts)[np.newaxis, :]
           for f, ts in zip(features, test_split)}

if roi:
    comb_r = reinsert_vertices(comb_r, roi_shape, roi_coords)
    split_r = {f: reinsert_vertices(s, roi_shape, roi_coords)
               for f, s in split_r.items()}

np.save(f'{save_dir}/comb-r_pca-{n_components}_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        comb_r)
np.save(f'{save_dir}/split-r_pca-{n_components}_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        split_r)

# Compute R^2 performance and save
comb_r2 = r2_score(test_data, test_comb)[np.newaxis, :]
split_r2 = {f: r2_score(test_data, ts)[np.newaxis, :]
            for f, ts in zip(features, test_split)}

if roi:
    comb_r2 = reinsert_vertices(comb_r2, roi_shape, roi_coords)
    split_r2 = {f: reinsert_vertices(s, roi_shape, roi_coords)
                for f, s in split_r2.items()}

np.save(f'{save_dir}/comb-r2_pca-{n_components}_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        comb_r2)
np.save(f'{save_dir}/split-r2_pca-{n_components}_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        split_r2)
        
print('\nFinished running banded ridge regression!')
print(f'Mean test r: {np.nanmean(comb_r):.03f}')
print(f'Mean test R2: {np.nanmean(comb_r2):.03f}')
end = time.time()
total_time = end - start
print(f"total_time: {total_time}")    
# TODO: SAVE ALPHAS?
# np.save(f'{save_dir}/split-r2_pca-{n_components}_align-{alignment}_{test_subject}_'
#         f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
#         split_r2)
