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


# hemisphere = 'lh'
# run = 1
# load HAcommon gifti files
# filename example: # sub-rid000041_task-life_acq-374vol_run-01_desc-HAcommon1.lh.gii
# gifti_list = glob.glob(os.path.join(gifti_dir, f'*_task-life_acq-*vol_run-01_desc-HAcommon1.lh.gii'))

# short_list= gifti_list[0:3]
# for gifti_fname in sorted(gifti_list):
for run in [1, 2, 3, 4]:
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
        cortical_vertices = ~medial_mask # boolean (true for non-medials, false for medials)
        cortical_coords = np.where(cortical_vertices)[0] # returns indices of non-medials
        print(f"number of non-medial vertices: {len(cortical_coords)}")
        print(f"all data shape (before removing medial): {all_data.shape}")
        cortical_data = all_data[..., cortical_coords]
        print(f"cortical data shape (after removing medial): {cortical_data.shape}")

        isc_result = np.zeros((len(subjects), gifti_data.shape[-1]))
        print(isc_result.shape)
        # isc across time
        subject_ids = range(len(subjects))


        isc_result = np.zeros((len(subjects), gifti_data.shape[-1]))  # 
        print(isc_result.shape)
        # isc across time
        subject_ids = np.arange(len(subjects))

        for subject_ind, subject in enumerate(subjects):
            target_subject = cortical_data[subject_ind, :, :]
            other_subjects = cortical_data[subject_ids != subject_ind, :, :] # TODO: print other_subjects is it (n-1)? is it excluding subject?
            other_avg = np.mean(other_subjects, axis=0)
            # other_subjects_concat = np.concatenate(other_subjects)
            # result_gifti = np.zeros((1,40962))# original shape
            isc = array_correlation(target_subject, other_avg, axis=0)
        # result_gifti = isc



            np.save(file = f'{isc_dir}/subjectwise-ISC_{subject}_HAcommon_run-{run}_{hemisphere}.npy', arr=isc) # save subject wise ISC
            suma_dir = '/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml'
            write_gifti_v2(
                isc.astype(np.float32),
                template_fn=os.path.join(suma_dir, f'{hemisphere}.pial.gii'), 
                output_fn=os.path.join(f'{isc_dir}/subjectwise-ISC_{subject}_HAcommon_run{run}_{hemisphere}.gii'))
            # save result_gifti 
            isc_result[subject_ind] = isc[0]
                # TODO: save to result_gifti
            print(f"run-{run} {hemisphere} {isc_result.shape}")
# write_gifti
# np.save(file= '/Volumes/life-encoding/scripts/sandbox_isc/group_isc_result.npy', arr = isc_result)