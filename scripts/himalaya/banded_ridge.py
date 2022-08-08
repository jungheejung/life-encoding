#!/usr/bin/env python3


import os, sys, shutil
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


# Assign/create directories
user_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f000fb6'
scratch_dir = os.path.join('/dartfs-hpc/scratch', user_dir)
data_dir = '/dartfs/rc/lab/D/DBIC/DBIC/life_data/'

# Create scratch directory if it doesn't exist
if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)

# Hyperalignment directory (/idata/DBIC/cara/life/pymvpa/)
hyper_dir = os.path.join(data_dir, 'hyperalign_mapper')

# Life fMRI data directory (/idata/DBIC/snastase/life)
fmri_dir = os.path.join(data_dir, 'life_dataset')

# Semantic model directory (/idata/DBIC/cara/w2v/w2v_features)
model_dir = os.path.join(data_dir, 'w2v_feature')

# Set up some hard-coded experiment variables
subjects = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006',
            'sub-rid000009', 'sub-rid000012', 'sub-rid000014',
            'sub-rid000017', 'sub-rid000019', 'sub-rid000024',
            'sub-rid000027', 'sub-rid000031', 'sub-rid000032',
            'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
            'sub-rid000037', 'sub-rid000038', 'sub-rid000041']

tr = 2.5
model_durs = {1: 369, 2: 341, 3: 372, 4: 406}
fmri_durs = {1: 374, 2: 346, 3: 377, 4: 412}
n_samples = np.sum(list(fmri_durs.values()))
n_vertices = 40962
n_medial = {'lh': 3486, 'rh': 3491}
n_blocks = 3 #20# 8 hr instead of 30 min ... break up n_vertices
model_ndim = 300
run_labels = [1, 2, 3, 4]
roi_json = os.path.join(user_dir, 'rois.json')

# Parameters for optional PCA
run_pca = False
n_components = 30

# Parameters from job submission script
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--alignment",
                    choices=['ws', 'aa', 'ha_test', 'ha_common'],
                    help="within-subject, anatomical, or hyperalignment")
parser.add_argument("--hemisphere", choices=['lh', 'rh'],
                    help="specify hemisphere")
parser.add_argument("-r", "--test-run", choices=[1, 2, 3, 4],
                    type=int, help="specify test run")
parser.add_argument("-s", "--test-subject", help="specify test subject")
parser.add_argument("-f", "--features", nargs="*", type=str,
                    default=['bg', 'actions', 'agents'],
                    help="specify one or more feature spaces")
parser.add_argument("--roi", default=None, type=str,
                    help="ROI label from rois.json")
args = parser.parse_args()

alignment = 'ha_common' #args.alignment # 'ws', 'aa', 'ha_test', 'ha_common'
hemisphere = 'lh' #args.hemisphere # 'lh' or 'rh'
test_run = 1 #args.test_run # 1, 2, 3, or 4
test_run_id = test_run - 1 # zero-indexed test run
train_runs = [r for r in run_labels if r != test_run] 
test_subject = 'sub-rid000005' #args.test_subject # e.g. 'sub-rid000005'
features = ['bg', 'actions', 'agents'] #args.features # e.g. ['bg']
roi = '0' #args.roi e.g. 'vt', '0'


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
                                       np.array(rois[hemisphere][roi]))
                         
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

        assert train_model.shape[1] == test_model.shape[1] == model_ndim
        train_model, test_model = model_pca(train_model,
                                            test_model,
                                            n_components)
        
        # Re-split training model into runs
        train_model = np.split(train_model, train_splits)

    # Horizontally stack lagged versions of the stimulus model
    lags = [1, 2, 3, 4]
    train_cats, train_durs = [], {}
    for r, train_run in zip(train_runs, train_model):
        train_cat = np.concatenate((train_run[lags[-1] - 1:-lags[0]],
                                    train_run[lags[-2] - 1:-lags[1]],
                                    train_run[lags[-3] - 1:-lags[2]],
                                    train_run[lags[-4] - 1:-lags[3]]), axis=1)
        
        # Z-score each training run
        train_cat = zscore(train_cat, axis=0)
        
        train_cats.append(train_cat)
        train_durs[r] = train_cat.shape[0]
        print(f"Train model run {r} shape:"
              f"\n\toriginal {train_run.shape} -> lagged {train_cat.shape}")
    train_model = np.concatenate(train_cats, axis=0)
    
    test_model = np.concatenate((test_model[lags[-1] - 1:-lags[0]],
                                 test_model[lags[-2] - 1:-lags[1]],
                                 test_model[lags[-3] - 1:-lags[2]],
                                 test_model[lags[-4] - 1:-lags[3]]), axis=1)
    
    # Z-score each model run separately
    test_model = zscore(test_model, axis=0)
    
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
                               f'{hemisphere}.gii')))[0, 3:-7, :]
            
            else:
                run_data = read_gifti(os.path.join(
                    fmri_dir, (f'{train_subject}_task-life_acq-'
                               f'{fmri_durs[train_run]}vol_run-'
                               f'{train_run:02d}_'
                               f'desc-HAcommon{test_run}.'
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
                       f'desc-HAcommon{test_run}.'
                       f'{hemisphere}.gii')))[0, 3:-7, :]
        
    else:
        test_data = read_gifti(os.path.join(
            fmri_dir, (f'{test_subject}_task-life_acq-'
                       f'{fmri_durs[test_run]}vol_run-'
                       f'{test_run:02d}_'
                       f'desc-HAcommon{test_run}.'
                       f'{hemisphere}.gii')))[0, 3:-6, :]

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
n_iter = 10
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

np.save(f'ridge-coef_align-{alignment}_{test_subject}_'
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

np.save(f'comb-pred_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        test_comb_save)
np.save(f'split-pred_align-{alignment}_{test_subject}_'
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

np.save(f'comb-r_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        comb_r)
np.save(f'split-r_align-{alignment}_{test_subject}_'
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

np.save(f'comb-r2_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        comb_r2)
np.save(f'split-r2_align-{alignment}_{test_subject}_'
        f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy',
        split_r2)
        
print('\nFinished running banded ridge regression!')
print(f'Mean test r: {np.nanmean(comb_r):.03f}')
print(f'Mean test R2: {np.nanmean(comb_r2):.03f}')
    
# TODO: SAVE ALPHAS?
