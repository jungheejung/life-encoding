# %% libraries
import numpy as np
import pandas as pd
import os
from os.path import join
# %% parameters

# parameters
suma_dir = '/Users/h/suma-fsaverage6'
main_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding'
output_dir = os.path.join(main_dir, 'results', 'himalaya', 'ha_common')
result = "split-r"
alignment = 'ha_common'
niml_dir = join(data_dir, 'niml')
gifti_dir = join(data_dir, 'life_dataset')
runs = [1, 2, 3, 4]
hemis = ['lh', 'rh']
fmri_durs = {1: 374, 2: 346, 3: 377, 4: 412}
n_samples = 1509
n_vertices = 40962
n_medial = {'lh': 3486, 'rh': 3491}
subjects = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006',
            'sub-rid000009', 'sub-rid000012', 'sub-rid000014',
            'sub-rid000017', 'sub-rid000019', 'sub-rid000024',
            'sub-rid000027', 'sub-rid000031', 'sub-rid000032',
            'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
            'sub-rid000037', 'sub-rid000038', 'sub-rid000041']
# %% 1. Fisher z transform per run per participant (np.arctanh)
# np.save(f"{output_dir}/{result}_align-{alignment}_sub-mean_run-{test_run}_hemi-{hemisphere}.npy", avg_run)
for hemisphere in hemis:
    avg_all = []
    hemi_t = []
    hemi_p = []
    for test_subject in subjects:
        stack_fisherz_run = []
        for test_run in runs:           
            run_data = np.load(f"{output_dir}/{result}_align-{alignment}_{test_subject}_run-{test_run}_hemi-{hemisphere}.npy")
            fisherz_run = np.tanh(run_data, axis = 0)
            stack_fisherz_run.append(fisherz_run)
# %% 2. average (z-transformed) correlations across runs: yields 18 maps (1 per subject)
        avg_run = np.mean(stack_fisherz_run)
        np.save({average z score map}, avg_run)
# %% 3. Scipy ttest_1samp to get t-value and p-value
    t, p = scipy.stats.ttest_1samp(avg_run, 0, 'two-sided')
    hemi_t.append(t)
    hemi_p.append(p)
# %% 4. cortical vertices
# Create mask of cortical vertices excluding medial wall    
medial_mask = np.load(os.path.join(niml_dir, f'fsaverage6_medial_{hemisphere}.npy'))
assert np.sum(medial_mask) == n_medial[hemisphere]
cortical_vertices = ~medial_mask # boolean (true for non-medials, false for medials)
cortical_coords = np.where(cortical_vertices)[0] # returns indices of non-medials
print(f"number of non-medial vertices: {len(cortical_coords)}")
print(f"all data shape (before removing medial): {run_data.shape}")
cortical_data = run_data[..., cortical_coords]
print(f"cortical data shape (after removing medial): {cortical_data.shape}")
# %% 4-1. concatenate (np.hstack) the two hemispheres p-values (and exclude medial wall) prior to computing FDR (load in cortical_vertices.npy)
t_all = np.hstack(hemi_t)
p_all = np.hstack(hemi_p)
# NOTE: keep track of non-medial indices and hemisphere to stick back into dictionary
# %% 5. use statsmodels multipletests for FDR correction (‘bh’)
# FDR correction yields FDR-adjusted p-values (q-values)
# significant voxels

