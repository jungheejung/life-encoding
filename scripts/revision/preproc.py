# load nifti
# use nilearn
# grab fmriprep parameters
# grab beh onset
# nilearn clean image
# %%
import os
from nilearn import image, signal
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# %% ====================================================================
# 1. load image and confounds 
# =======================================================================
nii = image.load_img('/Users/h/Downloads/sub-0052_ses-01_task-alignvideo_acq-mb8_run-01_bold.nii.gz')
confounds = pd.read_csv('/Users/h/Downloads/sub-0052_ses-01_task-alignvideo_acq-mb8_run-1_desc-confounds_timeseries.tsv', sep='\t')
confound_columns = ['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

confounds_selected = confounds[confound_columns]
confounds_matrix = confounds_selected.fillna(0).values
dtseries = nib.load('/Users/h/Downloads/sub-0052_ses-01_task-alignvideo_acq-mb8_run-1_space-fsLR_den-91k_bold.dtseries.nii')
tr = dtseries.header.get_axis(0).step 
cifti_data = dtseries.get_fdata()
# =======================================================================
# 2. clean image
# =======================================================================
clean_cifti = signal.clean(cifti_data,
      confounds=confounds_matrix,
      standardize=True,
      detrend=True,
      high_pass=1/128,
      t_r=0.46)
# %% ====================================================================
# 3. slice images based on video length
# =======================================================================
events = pd.read_csv('/Users/h/Downloads/sub-0052_ses-01_task-alignvideo_acq-mb8_run-01_events.tsv', sep='\t')
events
# %%
events[events['trial_type'] == 'video']['onset']
events[events['trial_type'] == 'video']['duration']

# snip out brain data based on onset and duration and 
# name the niftifiles acoording to the stim_file column
video_events = events[events['trial_type'] == 'video'].copy()

for idx, row in video_events.iterrows():
    onset = row['onset']  # in seconds
    duration = row['duration']  # in seconds
    stim_file = row['stim_file'] if 'stim_file' in row else f"video_{idx}"
    
    # Convert time to TRs (volumes)
    onset_tr = int(np.round(onset / tr))
    duration_tr = int(np.round(duration / tr))
    end_tr = onset_tr + duration_tr

        # Check bounds
    if onset_tr < 0:
        print(f"Warning: onset_tr {onset_tr} < 0 for {stim_file}")
        onset_tr = 0
    if end_tr > cifti_data.shape[0]:
        print(f"Warning: end_tr {end_tr} > data length {cifti_data.shape[0]} for {stim_file}")
        end_tr = cifti_data.shape[0]

    # Extract the segment
    segment_data = cifti_data[onset_tr:end_tr, :]
    
    print(f"Video: {stim_file}")
    print(f"  Onset: {onset}s (TR {onset_tr}) | Duration: {duration}s ({duration_tr} TRs)")
    print(f"  Extracted shape: {segment_data.shape}")
    
    # Create filename from stim_file
    if 'stim_file' in row and pd.notna(row['stim_file']):
        # Clean up the stimulus filename for use as output filename
        base_name = Path(row['stim_file']).stem  # Remove extension
        # Replace problematic characters
        safe_name = base_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        output_filename = f"{safe_name}.dtseries.nii"
    else:
        output_filename = f"video_segment_{idx:03d}.dtseries.nii"
    
        # Create new CIFTI header for the segment
    # Copy the original header but update the time axis
    new_header = dtseries.header.copy()
    
    # Update the time axis (axis 0) for the new duration
    time_axis = new_header.get_axis(0)
    new_time_axis = nib.cifti2.SeriesAxis(
        start=0,  # Start at 0 for the segment
        step=time_axis.step,  # Keep same TR
        size=segment_data.shape[0]  # New number of timepoints
    )
    # new_header.set_axis(0, new_time_axis)
    brain_axis = dtseries.header.get_axis(1) 
    new_header = nib.cifti2.Cifti2Header.from_axes((new_time_axis, brain_axis))
    
    # Create new CIFTI image
    segment_cifti = nib.Cifti2Image(
        segment_data,
        header=new_header,
        nifti_header=dtseries.nifti_header
    )

    # Save the segment
    output_dir = '/Users/h/Downloads'
    output_path = os.path.join(output_dir, output_filename)
    nib.save(segment_cifti, output_path)
    print(f"  Saved: {output_path}")
    print()
