#!/usr/bin/env python
"""
calculate intersubject correlation (ISC) per run, per hemisphere, and per align
___
    input from slurm batch script: hemisphere, modality
    this code loops through each run
    output: 
        - saves ISC per run (will later be useful for forward encoding noise ceiling estimation)
        - saves ISC averaged across runs for visualization purposes


"""
# %% load library ___________________________________________
import os, glob
from os.path import join
from scipy.stats import zscore
from scipy.stats import pearsonr

import sys
import nibabel as nib
import numpy as np
from matplotlib import colors

__author__ = "Heejung Jung, Same Nastase"
__copyright__ = "life-encoding"
__credits__ = [] # people who reported bug fixes, made suggestions, etc. but did not actually write the code.
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Heejung Jung"
__email__ = "heejung.jung@colorado.edu"
__status__ = "Development" 


# %% functions ___________________________________________
def read_gifti(gifti_fn):
    gii = nib.load(gifti_fn)
    data = np.vstack([da.data[np.newaxis, :]
                      for da in gii.darrays])
    return data

def write_gifti(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    for row in data:
        gda = nib.gifti.GiftiDataArray(row)
        gii.add_gifti_data_array(gda)
    nib.gifti.giftiio.write(gii, output_fn)

# %% parameters
align = sys.argv[1]
hemisphere = sys.argv[2]
align = 'ha_common'
hemisphere = 'lh'
# %% directories ___________________________________________
data_dir = '/Volumes/life_data'
niml_dir = join(data_dir, 'niml')
gifti_dir = join(data_dir, 'life_dataset')
suma_dir = '/Users/h/suma-fsaverage6'
output_dir = join('/Volumes/life-encoding', 'scripts', 'isc')
runs = [1, 2, 3, 4]
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

# %%

for run in runs:
    # 1. load data
    per_run = []
    for subject in subjects:
        if align == 'aa':
            # sub-rid000005_task-life_acq-346vol_run-02.lh.tproject
            fname = join(gifti_dir, f'{subject}_task-life_acq-{fmri_durs[run]}vol_run-0{run}.{hemisphere}.tproject.gii')
        elif align == 'ha_test':
            # sub-rid000005_task-life_acq-374vol_run-01_desc-HAtestR1rid000001.lh
            fname = join(gifti_dir, f'{subject}_task-life_acq-{fmri_durs[run]}vol_run-0{run}_desc-HAtestR{run}rid000005.{hemisphere}.gii')
        elif align == 'ha_common':
            # sub-rid000005_task-life_acq-374vol_run-01_desc-HAcommon1.lh
            fname = join(gifti_dir, f'{subject}_task-life_acq-{fmri_durs[run]}vol_run-0{run}_desc-HAcommon{run}.{hemisphere}.gii')

        print(fname)
        fdata = read_gifti(fname)
        per_run.append(fdata)
        print(f'individual gifti data shape: {fdata.shape}')
    all_fdata = np.squeeze(np.array(per_run))
    print(f'stacked data shape: {all_fdata.shape}')

    # 1-1. as a manipulation check, see how many nans we have per giftifile
    np.sum(np.all(np.isnan(np.squeeze(all_fdata[3])), axis=0))

    # 1-2. Create mask of cortical vertices excluding medial wall    
    medial_mask = np.load(os.path.join(data_dir, 'niml',
                                    f'fsaverage6_medial_{hemisphere}.npy'))
    assert np.sum(medial_mask) == n_medial[hemisphere]
    cortical_vertices = ~medial_mask # boolean (true for non-medials, false for medials)
    cortical_coords = np.where(cortical_vertices)[0] # returns indices of non-medials
    print(f"number of non-medial vertices: {len(cortical_coords)}")
    print(f"all data shape (before removing medial): {all_fdata.shape}")
    cortical_data = all_fdata[..., cortical_coords]
    print(f"cortical data shape (after removing medial): {cortical_data.shape}")

    # 1-3. create empty isc array
    isc_result = np.zeros((len(subjects), fdata.shape[-1]))
    print(isc_result.shape)
    subject_ids = np.arange(len(subjects))

    # 2. ISC calculation 
    for subject_ind, subject in enumerate(subjects):
        target_subject = cortical_data[subject_ind, :, :]
        other_subjects = cortical_data[subject_ids != subject_ind, :, :] # TODO: print other_subjects is it (n-1)? is it excluding subject?
        print(f"other subject size: , {other_subjects.shape}")
        other_avg = np.mean(other_subjects, axis=0)

        result_gifti = np.zeros((1,40962)) # original shape
        for i, voxel in enumerate(cortical_coords): 
            left_out_voxel = target_subject[:, i]
            other_avg_voxel = other_avg[:, i] 
            isc = pearsonr(left_out_voxel, other_avg_voxel)[0]  # get r-value from pearsonr
            # if isc and cortical_data[-1] shape match
            result_gifti[0, voxel] = isc # cortical_coords and cortical_data need to have matching dimensions

        # subjectwise ISC (numpy & gifti)
        np.save(file = join(output_dir, f"subjectwise-ISC_align-{align}_hemi-{hemisphere}_run-{run:02d}_{subject}.npy"), 
        arr = result_gifti) # save subject wise ISC
        write_gifti(result_gifti.astype(float),
        template_fn = join(suma_dir, f'{hemisphere}.pial.gii'), 
        output_fn = join(output_dir,f'subjectwise-ISC_align-{align}_hemi-{hemisphere}_run-{run:02d}_{subject}.gii'))
        isc_result[subject_ind] = result_gifti[0]
        # TODO: save to result_gifti
    # runwise ISC (numpy & average gifti)
    np.save(file= join(output_dir,f'runwise-ISC_align-{align}_run-{run:02d}_hemi-{hemisphere}.npy'), arr = isc_result)
    mean_runwise_isc = np.mean(isc_result, axis = 0)
    write_gifti(mean_runwise_isc.astype(float),
        template_fn = join(suma_dir, f'{hemisphere}.pial.gii'), 
        output_fn = join(output_dir,f'groupaverage_runwise-ISC_align-{align}_run-{run:02d}_hemi-{hemisphere}.gii'))



# load all 4 npy arrays and average them for visualization purposes
# total_run = np.array([], dtype=np.int64).reshape(len(subject_ids),40962)
total_run = np.empty((len(subject_ids),40962))
for run in runs:
    r = np.load(file= join(output_dir,f'runwise-ISC_align-{align}_run-{run:02d}_hemi-{hemisphere}.npy'))
    np.stack((total_run, r))
    mean_isc = np.mean(total_run, axis = 0)
    write_gifti(mean_isc.astype(float),
        template_fn = join(suma_dir, f'{hemisphere}.pial.gii'), 
        output_fn = join(output_dir,f'groupaverage-ISC_align-{align}_hemi-{hemisphere}.gii'))



# %%
