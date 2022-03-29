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
from himalaya.scoring import correlation_score
import nibabel as nib
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
n_proc = 32 # how many cores do we have?
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
parser.add_argument("-a", "--alignment", choices=['ws', 'aa', 'ha'],
                    help="within-subject, anatomical, or hyperalignment")
parser.add_argument("--hemisphere", choices=['lh', 'rh'],
                    help="specify hemisphere")
parser.add_argument("-r", "--test-run", choices=[1, 2, 3, 4],
                    type=int, help="specify test run")
parser.add_argument("-s", "--test-subject", help="specify test subject")
parser.add_argument("--roi", default=None,
                    help="ROI label from rois.json")
parser.add_argument("-f", "--features", nargs="*", type=str,
                    default=['bg', 'actions', 'agents'],
                    help="specify one or more feature spaces")
args = parser.parse_args()

alignment = 'ws' #args.alignment # 'ws', 'aa', 'ha'
hemisphere = 'lh' #args.hemisphere # 'lh' or 'rh'
test_run = 1 #args.test_run # 1, 2, 3, or 4
test_run_id = test_run - 1 # zero-indexed test run
train_runs = [r for r in run_labels if r != test_run] 
test_subject = 'sub-rid000005' #args.test_subject # e.g. 'sub-rid000005'
roi = 'vt' #args.roi e.g. 'vt'
features = ['bg', 'actions', 'agents'] #args.features # e.g. ['bg']


# Create mask of cortical vertices excluding medial wall    
medial_mask = np.load(os.path.join(data_dir, 'niml',
                                   f'fsaverage6_medial_{hemisphere}.npy'))
assert np.sum(medial_mask) == n_medial[hemisphere]
cortical_vertices = ~medial_mask

# Load ROIs and ensure ROI vertices are within cortical mask
if roi:
    with open(roi_json, 'r') as f:
        rois = json.load(f)
    
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
    print(f"Loading anatomically-aligned data for {test_subject}")
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
    print(f"Loading hyperaligned data for {test_subject}")
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
                               f'{train_run:02d}_'
                               f'desc-HAtestR{test_run}'
                               f"{test_subject.split('-')[1]}."
                               f'{hemisphere}.gii')))[3:-7, :]
            
            else:
                run_data = read_gifti(os.path.join(
                    fmri_dir, (f'{train_subject}_task-life_acq-'
                               f'{fmri_durs[train_run]}vol_run-'
                               f'{train_run:02d}_'
                               f'desc-HAtestR{test_run}'
                               f"{test_subject.split('-')[1]}."
                               f'{hemisphere}.gii')))[3:-6, :]

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
                       f'{train_run:02d}_'
                       f'desc-HAtestR{test_run}'
                       f"{test_subject.split('-')[1]}."
                       f'{hemisphere}.gii')))[3:-7, :]
        
    else:
        test_data = read_gifti(os.path.join(
            fmri_dir, (f'{test_subject}_task-life_acq-'
                       f'{fmri_durs[test_run]}vol_run-'
                       f'{train_run:02d}_'
                       f'desc-HAtestR{test_run}'
                       f"{test_subject.split('-')[1]}."
                       f'{hemisphere}.gii')))[3:-6, :]

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
print("Finished fitting banded ridge model")
print(f"Weight matrix shape: {ridge.coef_.shape}")

# Use the fitted ridge model to predict left-out test run
print("Predicting left-out test run")
test_comb = ridge.predict(test_bands)
test_split = ridge.predict(test_bands, split=True)
print(f"Predicted test run {test_run} shape: {test_pred.shape}")

test_score = correlation_score(test_comb, test_data)

# =================================================================





def get_ha_common_data(test_subject, mappers, test_run, train_runs, hemi):
    train_p = [x for x in subjects if x != test_subject]
    print("\n4. hyperalignment common data")
    print(
        'Loading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_subject))
    train_resp = []
    for run in train_runs:
        avg = []
        for participant in train_p:
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # UNCOMMENT LATER -
            # if run == 4:
            #     resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            #         participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            # else:
            #     resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            #         participant, tr_fmri[run], run, hemi))).samples[4:-4, :]
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]
            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # DELETE LATER -
            # resp = resp[:, cortical_vertices[hemi] == 1]

            resp = resp[:, selected_node]
            #
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    if test_run == 4:
        test_resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_subject, tr_fmri[test_run], test_run, hemi))).samples[4:-5, :]
    else:
        test_resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_subject, tr_fmri[test_run], test_run, hemi))).samples[4:-4, :]
    mv.zscore(test_resp, chunks_attr=None)
    test_resp = mappers[participant].forward(test_resp)
    # test_resp = test_resp[:, cortical_vertices[hemi] == 1]
    test_resp = test_resp[:, selected_node]
    mv.zscore(test_resp, chunks_attr=None)

    print('test', test_run, test_resp.shape)

    return train_resp, test_resp


def get_ha_testsubj_data(test_subject, mappers, test_run, train_runs, hemi):
    train_p = [x for x in subjects if x != test_subject]
    print("\n4. hyperalignment test subject data")
    print(
        'Loading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_subject))
    train_resp = []
    for run in train_runs:
        avg = []
        for participant in train_p:
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # UNCOMMENT LATER -
            # if run == 4:
            #     resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            #         participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            # else:
            #     resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            #         participant, tr_fmri[run], run, hemi))).samples[4:-4, :]
            # # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # DELETE LATER -
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-5, :]
            else:
                resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
                    participant, tr_fmri[run], run, hemi))).samples[4:-4, :]
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            mv.zscore(resp, chunks_attr=None)
            resp = mappers[test_subject].reverse(resp)
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # DELETE LATER -
            # resp = resp[:, cortical_vertices[hemi] == 1]

            resp = resp[:, selected_node]
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            mv.zscore(resp, chunks_attr=None)


            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('train', run, avg.shape)
        train_resp.append(avg)

    # if test_run == 4:
    #     test_resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
    #         test_subject, tr_fmri[test_run], test_run, hemi))).samples[4:-5, cortical_vertices[hemi] == 1]
    # else:
    #     test_resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
    #         test_subject, tr_fmri[test_run], test_run, hemi))).samples[4:-4, cortical_vertices[hemi] == 1]

    if test_run == 4:
        test_resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_subject, tr_fmri[test_run], test_run, hemi))).samples[4:-5, selected_node]
    else:
        test_resp = mv.gifti_dataset(os.path.join(fmri_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_subject, tr_fmri[test_run], test_run, hemi))).samples[4:-4, selected_node]

    mv.zscore(test_resp, chunks_attr=None)

    print('test', test_run, test_resp.shape)

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
    print(proc_stdout)


# 2. Load data _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

# 2-2) load visual or narrative feature data _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
# stimfile1 = 'bg'
X1train_stim, X1test_stim = get_visual_stim_for_fold(
    '{0}_{1}'.format(model, stimfile1), test_run, train_runs)
# stimfile2 = 'actions'
X2train_stim, X2test_stim = get_visual_stim_for_fold(
    '{0}_{1}'.format(model, stimfile2), test_run, train_runs)
# stimfile1 = 'agents'
X3train_stim, X3test_stim = get_visual_stim_for_fold(
    '{0}_{1}'.format(model, stimfile3), test_run, train_runs)

    # 2-3) load fMRI data __________________________________________________
if align == 'ws':
    Ytrain_unconcat, Ytest = get_ws_data(
        test_subject, test_run, train_runs, hemi)
elif align == 'aa':
    Ytrain_unconcat, Ytest = get_aa_data(
        test_subject, test_run, train_runs, hemi)
else:
    print('\nLoading hyperaligned mappers...')
    ### TODO: apply hyperalignment mappers in separate script (w/ pymvpa env)
    # and load in already-mapped data as GIfTI or np.ndarrays
    #mappers = mv.h5load(os.path.join(
    #    hyper_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(hemi, test_run)))
    #if align == 'ha_common':
    #    Ytrain_unconcat, Ytest = get_ha_common_data(
    #        test_subject, mappers, test_run, train_runs, hemi)
    #elif align == 'ha_testsubj':
    #    Ytrain_unconcat, Ytest = get_ha_common_data(
    #        test_subject, mappers, test_run, train_runs, hemi)

# delete later: ONLY ONE NODE
# print(Ytrain_unconcat.shape, "Y train unconcatenate shape")

# 2-4) concatenate 3 runs ______________________________________________
X1train = np.concatenate(X1train_stim)
X2train = np.concatenate(X2train_stim)
X3train = np.concatenate(X3train_stim)
Ytrain = np.concatenate(Ytrain_unconcat)

# 2-5) Print for JOB LOG ___________________________________________________
print('\nShape of training and testing set')
print(X1train.shape, "X1train shape") # ((1073, 120), 'X1train')
print(X2train.shape, "X2train shape") # ((1073, 120), 'X2train')
print(X3train.shape, "X3train shape") # ((1073, 120), 'X3train')
print(X1test_stim.shape, "X1test stim") # ((403, 120), 'X1test_stim')
print(X2test_stim.shape, "X2test stim") # ((403, 120), 'X2test_stim')
print(X3test_stim.shape, "X3test stim") # ((403, 120), 'X3test_stim')
print(Ytest.shape, "Ytest") # ((403, 3), 'Ytest')
print(Ytrain.shape, "Ytrain") # ((1073, 3), 'Ytrain')


# 3. [ banded ridge ] alpha and ratios _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
alphas = np.logspace(0, 5, 20)
#alphas = np.logspace(0, 3, 20)  # commented out for current analysis
# alphas = [18.33] # currently using for our quick analysis. For the main analysis, use the full logspace as above
ratios = np.logspace(-2, 2, 25)
print("\nalphas: {0}".format(alphas))
print("\nRatios: {0}".format(ratios))

train_id = np.arange(X1train.shape[0])
dur1, dur2, dur3 = tr_movie[train_runs[0]] - \
    3, tr_movie[train_runs[1]] - 3, tr_movie[train_runs[2]] - 3
print("\ndur1: {0}".format(dur1))
print("\ndur2: {0}".format(dur2))
print("\ndur3: {0}".format(dur3))

# 4. [ banded ridge ] setting up loro and priors _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

loro1 = [(train_id[:dur1 + dur2], train_id[dur1 + dur2:]),
         (np.concatenate((train_id[:dur1], train_id[dur1 + dur2:]), axis=0),
          train_id[dur1:dur1 + dur2]),
         (train_id[dur1:], train_id[:dur1])]

X1_prior = spatial_priors.SphericalPrior(X1train, hyparams=ratios)
X2_prior = spatial_priors.SphericalPrior(X2train, hyparams=ratios)
X3_prior = spatial_priors.SphericalPrior(X3train, hyparams=ratios)

temporal_prior = temporal_priors.SphericalPrior(delays=[0])  # no delays


# 5. [ banded ridge ] banded ridge regression _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

fit_banded_polar = models.estimate_stem_wmvnp([X1train, X2train, X3train], Ytrain,
                                              [X1test_stim, X2test_stim,
                                                  X3test_stim], Ytest,
                                              feature_priors=[
                                                  X1_prior, X2_prior, X3_prior],
                                              temporal_prior=temporal_prior,
                                              ridges=alphas, folds=loro1,
                                              performance=True, weights=True, verbosity=False)

voxelwise_optimal_hyperparameters = fit_banded_polar['optima']
print('\nVoxelwise optimal hyperparameter shape: {0}'.format(voxelwise_optimal_hyperparameters.shape)) # (23, 5)

# Faster conversion of kernel weights to primal weights via matrix multiplication
# each vector (new_alphas, lamda_ones, lamda_twos) contains v number of entries (e.g. voxels)
new_alphas = voxelwise_optimal_hyperparameters[:, -1] # 4
lambda_ones = voxelwise_optimal_hyperparameters[:, 1]
lambda_twos = voxelwise_optimal_hyperparameters[:, 2]
lambda_threes = voxelwise_optimal_hyperparameters[:, 3]
print("\nhyperparameter_0: {0}".format(voxelwise_optimal_hyperparameters[:, 0]))
# TODO: what is voxelwise_optimal_hyperparameters[:, 0]?

directory = os.path.join(scratch_dir, 'PCA_tikreg-loro_fullrange-10000-ROI','{0}/{1}/{2}_{3}_{4}/leftout_run_{5}'.format(align, model, stimfile1, stimfile2, stimfile3, test_run), test_subject, hemi)
hyperparameter_file = os.path.join(directory, 'hyperparameter.npy') 
with open(hyperparameter_file, 'wb') as f:
    np.save(f, voxelwise_optimal_hyperparameters)


# 6. [ banded ridge ] calculating primal weights from kernel weights _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

kernel_weights = fit_banded_polar['weights']
# 6-1. calculate the beta weights from primal weights ______________________
weights_x1 = np.linalg.multi_dot(
    [X1train.T, kernel_weights, np.diag(new_alphas), np.diag(lambda_ones**-2)])
weights_x2 = np.linalg.multi_dot(
    [X2train.T, kernel_weights, np.diag(new_alphas), np.diag(lambda_twos**-2)])
weights_x3 = np.linalg.multi_dot(
    [X3train.T, kernel_weights, np.diag(new_alphas), np.diag(lambda_threes**-2)])

weights_joint = np.vstack((weights_x1, weights_x2, weights_x3))
teststim_joint = np.hstack((X1test_stim, X2test_stim, X3test_stim))
print("Feature 1 weight shape: {0}".format(weights_x1.shape)) #(120, 1)
print("Feature 2 weight shape: {0}".format(weights_x2.shape)) #(120, 1)
print("Feature 3 weight shape: {0}".format(weights_x3.shape)) #(120, 1)
print("weights type: {0}".format(type(weights_x1)))
print("Joint weights shape: {0}".format(weights_joint.shape))
print("Joint stim shape: {0}".format(teststim_joint.shape))

# 6-2. calculate the estimated Y based on the primal weights _________________
estimated_y1 = np.linalg.multi_dot([X1test_stim, weights_x1]) # (369, 120) * (120, 5)
estimated_y2 = np.linalg.multi_dot([X2test_stim, weights_x2])
estimated_y3 = np.linalg.multi_dot([X3test_stim, weights_x3])
estimated_ytotal = np.linalg.multi_dot([teststim_joint, weights_joint]) # (369, 360) * (360, 5)

directory = os.path.join(scratch_dir, 'PCA_tikreg-loro_fullrange-10000-ROI',
                         '{0}/{1}/{2}_{3}_{4}/leftout_run_{5}'.format(align, model, stimfile1, stimfile2, stimfile3, test_run), test_subject, hemi)
if not os.path.exists(directory):
    os.makedirs(directory)

print("\nsave directory: {0}".format(directory))

print("\nalpha shape: {0}".format(new_alphas.shape))
print("alpha type: {0}".format(type(new_alphas)))

# 6-4. save alpha_________________________________________________
# corr_shape = n_vertices - n_medial[hemi]
# outhyperparam = np.zeros(
#     (corr_shape + med_wall_ind.shape[0]), dtype=new_alphas.dtype)
#outhyperparam = np.zeros(node_range.shape[0], dtype=new_alphas.dtype)
print("\n6-4. alphas")

ind_nonmedial = np.array(selected_node) # insert nonmedial index
print("ind_nonmedial: {0}".format(ind_nonmedial))
ind_medial = np.array(medial_node) # insert medial index
print("ind_medial: {0}".format(ind_medial))
append_zero = np.zeros(len(medial_node)) # insert medial = 0
alpha_nonmedial = np.array(new_alphas) # insert nonmedial alpha
print("append_zero: {0}".format(append_zero))
print("alpha_nonmedial: {0}".format(alpha_nonmedial))
weight_x1_nonmedial = np.array(weights_x1)
weight_x2_nonmedial = np.array(weights_x2)
weight_x3_nonmedial = np.array(weights_x3) # (120, 5)
weights_joint_nonmedial = np.array(weights_joint)

if len(medial_node) != 0:
    index_chunk = np.concatenate((ind_nonmedial,ind_medial), axis = None)
    alpha_value = np.concatenate((alpha_nonmedial,append_zero),axis = None)
else:
    index_chunk = ind_nonmedial
    alpha_value = alpha_nonmedial

zipped_alphas = zip(index_chunk.astype(float), alpha_value.astype(float))
sorted_alphas = sorted(zipped_alphas)


#outhyperparam[outhyperparam] = new_alphas

# 6-4. save alpha
# https://stackoverflow.com/questions/64082441/save-a-list-of-tuples-to-a-file-and-read-it-again-as-a-list
alpha_savename = os.path.join(directory, 'hyperparam-alpha_{0}_model-{1}_align-{2}_foldshifted-{3}_hemi-{4}_range-{5}.json'.format(
        test_subject, model, align,  test_run, hemi,roi))
with open(alpha_savename, 'w') as f:
     json.dump(sorted_alphas, f)

# 6-3. save primal weights _________________________________________________
# [ ] TO DO: primal weights. make sure to grab the shape and create numpy zeros of that shape
# [ ] save tuple index and numpy array
#
# weights nonmedial shape: 120
# length of medial nodes: 6
# weight_zero shape: (120, 14)
# weightx1_value shape: (20, 120)
print("\n6-3.save primal weights")
print("weights nonmedial shape: {0}".format(weight_x3_nonmedial.shape[0])) #(120, 5)
print("length of medial nodes: {0}".format(len(medial_node)))

if len(medial_node) != 0:
    index_chunk = np.concatenate((ind_nonmedial,ind_medial), axis = None)
    weight_zero = np.zeros((weight_x3_nonmedial.shape[0], len(medial_node))) # 120, 6# insert medial = 0
    weight_jzero = np.zeros((weight_x3_nonmedial.shape[0]*3, len(medial_node)))
    print("weight_nonmedial shape: {0}".format(weight_x3_nonmedial.shape)) # 120, 14
    print("weight_zero shape: {0}".format(weight_zero.shape)) # 120, 14
    print("weight_jzero shape: {0}".format(weight_jzero.shape)) # 120, 14
    weightx1_value = np.transpose(np.hstack((weight_x1_nonmedial,weight_zero)))
    weightx2_value = np.transpose(np.hstack((weight_x2_nonmedial,weight_zero)))
    weightx3_value = np.transpose(np.hstack((weight_x3_nonmedial,weight_zero)))
    weightj_value  = np.transpose(np.hstack((weights_joint_nonmedial,weight_jzero))) #360, 14, 360, 5
    print("weightx1_value shape: {0}".format(weightx1_value.shape)) # 600
    # weight_zero shape: (120, 14)
# weightx1_value shape: (20, 120)
elif len(medial_node) == 0:
    index_chunk    = ind_nonmedial
    weightx1_value = np.transpose(weight_x1_nonmedial)
    weightx2_value = np.transpose(weight_x2_nonmedial)
    weightx3_value = np.transpose(weight_x3_nonmedial)
    weightj_value  = np.transpose(weights_joint_nonmedial)
    print("weightx1_value shape: {0}".format(weightx1_value.shape))

#numpy transpose weight_x3_nonmedial (120, 5) -> (5, 120)
w_x1_dict = {e: weightx1_value[i] for i, e in enumerate(index_chunk)}
w_x2_dict = {e: weightx2_value[i] for i, e in enumerate(index_chunk)}
w_x3_dict = {e: weightx3_value[i] for i, e in enumerate(index_chunk)}
w_xj_dict = {e: weightj_value[i] for i, e in enumerate(index_chunk)}

print("type of weight x1 dictionary: {0}".format(type(w_x1_dict)))
# zipped_weightx1 = zip(index_chunk.astype(float), weightx1_value.astype(float))
# sorted_weightx1 = sorted(zipped_weightx1)
# zipped_weightx2 = zip(index_chunk.astype(float), weightx2_value.astype(float))
# sorted_weightx2 = sorted(zipped_weightx2)
# zipped_weightx3 = zip(index_chunk.astype(float), weightx3_value.astype(float))
# sorted_weightx3 = sorted(zipped_weightx3)
# zipped_weightj = zip(index_chunk.astype(float), weightj_value.astype(float))
# sorted_weightj = sorted(zipped_weightj)

weightx1_savename = os.path.join(directory, 'primal-weights_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}.json'.format(
        test_subject, model, align, stimfile1, test_run, hemi,roi))
weightx2_savename = os.path.join(directory, 'primal-weights_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}.json'.format(
        test_subject, model, align, stimfile2, test_run, hemi,roi))
weightx3_savename = os.path.join(directory, 'primal-weights_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}.json'.format(
        test_subject, model, align, stimfile3, test_run, hemi,roi))
weightj_savename = os.path.join(directory, 'primal-weights_{0}_model-{1}_align-{2}_feature-total_foldshifted-{3}_hemi-{4}_range-{5}.json'.format(
        test_subject, model, align, test_run, hemi, roi))

with open(weightx1_savename, 'w') as f:
     json.dump(w_x1_dict, f, sort_keys=True, indent=4, cls=NumpyEncoder)
with open(weightx2_savename, 'w') as f:
     json.dump(w_x2_dict, f, sort_keys=True, indent=4, cls=NumpyEncoder)
with open(weightx3_savename, 'w') as f:
     json.dump(w_x3_dict, f, sort_keys=True, indent=4, cls=NumpyEncoder)
with open(weightj_savename, 'w') as f:
     json.dump(w_xj_dict, f, sort_keys=True, indent=4, cls=NumpyEncoder)

# later use: to load this "jsonify"-ed numpy array
# obj_text = codecs.open(w_xj_dict, 'r', encoding='utf-8').read()
# b_new = json.loads(obj_text)
# a_new = np.array(b_new)

# 7. [ banded ridge ] correlation coefficient between actual Y and estimated Y _ _ _ _ _ _ _

actual_df = pd.DataFrame(data=Ytest)
estimated_y1_df = pd.DataFrame(data=estimated_y1)
estimated_y2_df = pd.DataFrame(data=estimated_y2)
estimated_y3_df = pd.DataFrame(data=estimated_y3)
estimated_ytotal_df = pd.DataFrame(data=estimated_ytotal)

corr_x1 = pd.DataFrame.corrwith(
    estimated_y1_df, actual_df, axis=0, method='pearson')
corr_x2 = pd.DataFrame.corrwith(
    estimated_y2_df, actual_df, axis=0, method='pearson')
corr_x3 = pd.DataFrame.corrwith(
    estimated_y3_df, actual_df, axis=0, method='pearson')
corr_total = pd.DataFrame.corrwith(
    estimated_ytotal_df, actual_df, axis=0, method='pearson')


# 7-1. save files _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
print("7. correlation values")
corr_x1_nonmedial = corr_x1.to_numpy()
corr_x2_nonmedial = corr_x2.to_numpy()
corr_x3_nonmedial = corr_x3.to_numpy()
corr_t_nonmedial = corr_total.to_numpy()
print("corr_x1_nonmedial shape: {0}".format(corr_x1_nonmedial.shape))
print("corr_t_nonmedial shape: {0}".format(corr_t_nonmedial.shape))
if len(medial_node) != 0:
    index_chunk = np.concatenate((ind_nonmedial,ind_medial), axis = None)
    append_zero = np.zeros(len(medial_node))
    appendj_zero = np.zeros(len(medial_node)*3)
    # append_zero = np.zeros((corr_x1_nonmedial.shape[0], len(medial_node))) # insert medial = 0
    # appendj_zero = np.zeros((corr_t_nonmedial.shape[0], len(medial_node))) # insert medial = 0
    #index_chunk = np.concatenate((ind_nonmedial,ind_medial), axis = None)
    # alpha_value = np.concatenate((alpha_nonmedial,append_zero),axis = None)

    corrx1_value = np.concatenate((corr_x1_nonmedial,append_zero))
    corrx2_value = np.concatenate((corr_x2_nonmedial,append_zero))
    corrx3_value = np.concatenate((corr_x3_nonmedial,append_zero))
    corr_t_value = np.concatenate((corr_t_nonmedial,appendj_zero))
elif len(medial_node) == 0:
    index_chunk = ind_nonmedial
    corrx1_value = corr_x1_nonmedial
    corrx2_value = corr_x2_nonmedial
    corrx3_value = corr_x3_nonmedial
    corr_t_value = corr_t_nonmedial
    print("weightx1_value shape: {0}".format( weightx1_value.shape[0])) # (20, 120)

zipped_corrx1 = zip(index_chunk.astype(float), corrx1_value.astype(float))
zipped_corrx2 = zip(index_chunk.astype(float), corrx2_value.astype(float))
zipped_corrx3 = zip(index_chunk.astype(float), corrx3_value.astype(float))
zipped_corrjoint = zip(index_chunk.astype(float), corr_t_value.astype(float))

sorted_corrx1 = sorted(zipped_corrx1);
sorted_corrx2 = sorted(zipped_corrx2);
sorted_corrx3 = sorted(zipped_corrx3);
sorted_corrjoint = sorted(zipped_corrjoint);

corrx1_savename = os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}.json'.format(
        test_subject, model, align, stimfile1, test_run, hemi,roi))
corrx2_savename = os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}.json'.format(
        test_subject, model, align, stimfile2, test_run, hemi,roi))
corrx3_savename = os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}.json'.format(
        test_subject, model, align, stimfile3, test_run, hemi,roi))
corrt_savename = os.path.join(directory, 'corrcoef_{0}_model-{1}_align-{2}_feature-total_foldshifted-{3}_hemi-{4}_range-{5}.json'.format(
        test_subject, model, align, test_run, hemi, roi))

with open(corrx1_savename, 'w') as f:
     json.dump(sorted_corrx1, f)
with open(corrx2_savename, 'w') as f:
     json.dump(sorted_corrx2, f)
with open(corrx3_savename, 'w') as f:
     json.dump(sorted_corrx3, f)
with open(corrt_savename, 'w') as f:
     json.dump(sorted_corrjoint, f)


# # [ 8. r-squared ] _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
# y1_pred = r2_score(Ytest, estimated_y1)
# y2_pred = r2_score(Ytest, estimated_y2)
# y3_pred = r2_score(Ytest, estimated_y3)
# yt_pred = r2_score(Ytest, estimated_ytotal)
#
# print(y1_pred)
# print(y1_pred.shape)
# print(yt_pred)
# if len(medial_node) != 0:
#     index_chunk = np.concatenate((ind_nonmedial,ind_medial), axis = None)
#     append_zero = np.zeros(len(medial_node))
#     appendj_zero = np.zeros(len(medial_node)*3)
#     r2_y1 = np.concatenate((y1_pred,append_zero));    r2_y2 = np.concatenate((y2_pred,append_zero));    r2_y3 = np.concatenate((y3_pred,append_zero));    r2_yt = np.concatenate((yt_pred,appendj_zero))
# elif len(medial_node) == 0:
#     index_chunk = ind_nonmedial
#     r2_y1 = y1_pred;    r2_y2 = y2_pred;    r2_y3 = y3_pred;    r2_yt = yt_pred
#
# zipped_r2_y1 = zip(index_chunk.astype(float), r2_y1.astype(float))
# zipped_r2_y2 = zip(index_chunk.astype(float), r2_y2.astype(float))
# zipped_r2_y3 = zip(index_chunk.astype(float), r2_y3.astype(float))
# zipped_r2_yt = zip(index_chunk.astype(float), r2_yt.astype(float))
#
# sorted_r2_y1 = sorted(zipped_r2_y1)
# sorted_r2_y2 = sorted(zipped_r2_y2)
# sorted_r2_y3 = sorted(zipped_r2_y3)
# sorted_r2_yt = sorted(zipped_r2_yt)
#
# r2_feat1_fname = os.path.join(directory, 'rsquared_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}.json'.format(
#         test_subject, model, align, stimfile1, test_run, hemi,roi))
# r2_feat2_fname = os.path.join(directory, 'rsquared_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}.json'.format(
#         test_subject, model, align, stimfile2, test_run, hemi,roi))
# r2_feat3_fname = os.path.join(directory, 'rsquared_{0}_model-{1}_align-{2}_feature-{3}_foldshifted-{4}_hemi-{5}_range-{6}.json'.format(
#         test_subject, model, align, stimfile3, test_run, hemi,roi))
# r2_featt_fname = os.path.join(directory, 'rsquared_{0}_model-{1}_align-{2}_feature-total_foldshifted-{3}_hemi-{4}_range-{5}.json'.format(
#         test_subject, model, align, test_run, hemi, roi))
#
# with open(r2_feat1_fname, 'w') as f:
#      json.dump(sorted_r2_y1, f)
# with open(r2_feat2_fname, 'w') as f:
#      json.dump(sorted_r2_y2, f)
# with open(r2_feat3_fname, 'w') as f:
#      json.dump(sorted_r2_y3, f)
# with open(r2_featt_fname, 'w') as f:
#      json.dump(sorted_r2_yt, f)
## add medial wall back in
# copy files and remove files _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
subprocess_cmd('cp -rf /scratch/f0042x1/PCA_tikreg-loro_fullrange-10000-ROI/ /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results; rm -rf /scratch/f0042x1/*')

print("\nprocess complete")

