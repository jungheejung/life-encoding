import os, glob
from os.path import join
from scipy.stats import zscore
from scipy.stats import pearsonr

import sys
import nibabel as nib
import numpy as np
from matplotlib import colors


def read_gifti(gifti_fn):
    gii = nib.load(gifti_fn)
    data = np.vstack([da.data[np.newaxis, :]
                      for da in gii.darrays])
    return data

def write_gifti_v2(data, output_fn, template_fn):
    """Write a 1-D float32 array to a gifti file using template metadata."""
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    gda = nib.gifti.GiftiDataArray(data)
    gii.add_gifti_data_array(gda)
    gii.to_filename(output_fn)

# Array correlation function to avoid loops
def array_correlation(x, y, axis=0):
    # Center (de-mean) input variables
    x_demean = x - np.mean(x, axis=axis)
    y_demean = y - np.mean(y, axis=axis)

    # Compute summed product of centered variables
    numerator = np.sum(x_demean * y_demean, axis=axis)

    # Compute sum squared error
    denominator = np.sqrt(np.sum(x_demean ** 2, axis=axis) *
                          np.sum(y_demean ** 2, axis=axis))

    return numerator / denominator


data_dir = '/vast/labs/DBIC/datasets/Life/life_data'
isc_dir = '/dartfs/rc/lab/H/HaxbyLab/heejung/results/revision/isc'
suma_dir = '/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml'
niml_dir = join(data_dir, 'niml')
gifti_dir = join(data_dir, 'life_dataset')
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

# isc_result_all[run][hemisphere] = (n_subjects x n_vertices) array, full vertex count, medial wall zeroed
isc_result_all = {}

for run in runs:
    isc_result_all[run] = {}

    for hemisphere in ['lh', 'rh']:
        per_run = []
        for subject in subjects:
            gifti_fname = join(gifti_dir, f'{subject}_task-life_acq-{fmri_durs[run]}vol_run-0{run}_desc-HAcommon{run}.{hemisphere}.gii')
            print(gifti_fname)
            gifti_data = read_gifti(gifti_fname)
            per_run.append(gifti_data)
            print(f'individual gifti data shape: {gifti_data.shape}')
        all_data = np.squeeze(np.array(per_run))
        print(f'stacked data shape: {all_data.shape}')

        # Create mask of cortical vertices excluding medial wall
        medial_mask = np.load(os.path.join(data_dir, 'niml',
                                        f'fsaverage6_medial_{hemisphere}.npy'))
        assert np.sum(medial_mask) == n_medial[hemisphere]
        cortical_vertices = ~medial_mask  # boolean (true for non-medials, false for medials)
        cortical_coords = np.where(cortical_vertices)[0]  # indices of non-medials
        print(f"number of non-medial vertices: {len(cortical_coords)}")
        print(f"all data shape (before removing medial): {all_data.shape}")
        cortical_data = all_data[..., cortical_coords]
        print(f"cortical data shape (after removing medial): {cortical_data.shape}")

        # isc_result: full vertex count, so medial wall can be zero-filled per subject
        isc_result = np.zeros((len(subjects), n_vertices))
        print(isc_result.shape)

        subject_ids = np.arange(len(subjects))

        for subject_ind, subject in enumerate(subjects):
            target_subject = cortical_data[subject_ind, :, :]
            other_subjects = cortical_data[subject_ids != subject_ind, :, :]  # excludes target subject -> n-1
            other_avg = np.mean(other_subjects, axis=0)

            isc = array_correlation(target_subject, other_avg, axis=0)  # shape: (n_cortical,)

            # save subject-wise cortical-only ISC (.npy)
            np.save(file=f'{isc_dir}/subjectwise-ISC_{subject}_HAcommon_run-{run}_{hemisphere}.npy', arr=isc)

            # re-insert into full vertex space (medial wall = 0) for gifti + isc_result
            isc_full = np.zeros(n_vertices)
            isc_full[cortical_coords] = isc

            write_gifti_v2(
                isc_full.astype(np.float32),
                template_fn=os.path.join(suma_dir, f'{hemisphere}.pial.gii'),  # fixed: matches actual hemisphere
                output_fn=os.path.join(f'{isc_dir}/subjectwise-ISC_{subject}_HAcommon_run-{run}_{hemisphere}.gii'))  # fixed: dash added before {run} to match .npy naming

            isc_result[subject_ind] = isc_full  # fixed: full vertex-wise array, not isc[0]

            print(f"run-{run} {hemisphere} {isc_result.shape}")

        isc_result_all[run][hemisphere] = isc_result  # saved per run+hemisphere, not overwritten

# Optionally save the full group-level ISC array per run/hemisphere
for run in runs:
    for hemisphere in ['lh', 'rh']:
        np.save(
            file=f'{isc_dir}/group_isc_result_run-{run}_{hemisphere}.npy',
            arr=isc_result_all[run][hemisphere]
        )