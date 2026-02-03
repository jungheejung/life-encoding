# load nifti
# use nilearn
# grab fmriprep parameters
# grab beh onset
# nilearn clean image

# 
# %%
import os
import subprocess
from nilearn import image, signal
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import numpy as np

import sys
import argparse

# Use argparse to handle the subject parameter
parser = argparse.ArgumentParser()
parser.add_argument("--sub", type=str, help="Subject ID (e.g., sub-0052)")
args = parser.parse_args()

SUB = args.sub
print(f"Starting pipeline for {SUB}")

def convert_and_save_gifti(segment_data, base_name, gifti_dir):
    """
    Splits the data array into left and right hemispheres and saves
    each as a separate GIFTI file.
    """
    # Standard fsLR 91k vertices for splitting hemispheres
    n_left = 29696
    n_right = 29716

    # Split the data array
    left_data = segment_data[:n_left]
    right_data = segment_data[n_left:n_left+n_right]
    
    # Save Left Hemisphere as GIFTI
    left_gifti = nib.gifti.GiftiImage()
    # Correctly create the data array with intent and datatype
    left_da = nib.gifti.GiftiDataArray(left_data.astype(np.float32), intent='NIFTI_INTENT_NONE')
    left_gifti.add_gifti_data_array(left_da)
    left_path = gifti_dir / f'{base_stimfile}_lh.func.gii'
    nib.save(left_gifti, left_path)
    print(f"  Saved left GIFTI: {left_path}")

    # Save Right Hemisphere as GIFTI
    right_gifti = nib.gifti.GiftiImage()
    
    right_da = nib.gifti.GiftiDataArray(right_data.astype(np.float32), intent='NIFTI_INTENT_NONE')
    right_gifti.add_gifti_data_array(right_da)
    right_path = gifti_dir / f'{base_stimfile}_rh.func.gii'
    nib.save(right_gifti, right_path)
    print(f"  Saved right GIFTI: {right_path}")
# %% ====================================================================
# 1. load image and confounds 
# =======================================================================
SPACETOP_PATH = '/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/derivatives/ds005256-fmriprep'
EVENTS_PATH = '/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop/dartmouth'


# SUB = 'sub-0052'
# SES = 'ses-01'
# RUN = 'run-01'
# RUN_NUM = int(RUN.split('-')[-1])

# The list of specific videos/runs you want to extract

targets = [
    {"ses": "ses-01", "run": "run-01", "order": "02", "content": "wanderers"},
    {"ses": "ses-01", "run": "run-02", "order": "02", "content": "HB"},
    {"ses": "ses-01", "run": "run-03", "order": "01", "content": "huggingpets"},
    {"ses": "ses-01", "run": "run-03", "order": "04", "content": "dancewithdeath"},
    {"ses": "ses-01", "run": "run-04", "order": "02", "content": "angrygrandpa"},
    {"ses": "ses-02", "run": "run-02", "order": "03", "content": "menrunning"},
    {"ses": "ses-02", "run": "run-03", "order": "01", "content": "unefille"},
    {"ses": "ses-02", "run": "run-03", "order": "04", "content": "war"},
    {"ses": "ses-03", "run": "run-02", "order": "01", "content": "planetearth"},
    {"ses": "ses-03", "run": "run-02", "order": "03", "content": "heartstop"},
    {"ses": "ses-03", "run": "run-03", "order": "01", "content": "normativeprosocial2"},
    {"ses": "ses-04", "run": "run-01", "order": "02", "content": "gockskumara"}
]

# -----------------------------------------------------------------------------
# 1. preprocessing
# -----------------------------------------------------------------------------
for idx, target in enumerate(targets):
    SES = target['ses']
    RUN = target['run']
    CONTENT = target['content']
    # nii = image.load_img('/Users/h/Downloads/sub-0052_ses-01_task-alignvideo_acq-mb8_run-01_bold.nii.gz')
    confounds = pd.read_csv(f'{SPACETOP_PATH}/{SUB}/{SES}/func/{SUB}_{SES}_task-alignvideo_acq-mb8_{RUN}_desc-confounds_timeseries.tsv', sep='\t')

    confound_columns = ['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    confounds_selected = confounds[confound_columns]
    confounds_matrix = confounds_selected.fillna(0).values

    dtseries = nib.load(f'{SPACETOP_PATH}/{SUB}/{SES}/func/{SUB}_{SES}_task-alignvideo_acq-mb8_{RUN}_space-fsLR_den-91k_bold.dtseries.nii')
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
    events = pd.read_csv(f'{EVENTS_PATH}/{SUB}/{SES}/func/{SUB}_{SES}_task-alignvideo_acq-mb8_{RUN}_events.tsv', sep='\t')
    

    events[events['trial_type'] == 'video']['onset']
    events[events['trial_type'] == 'video']['duration']

    # snip out brain data based on onset and duration and 
    # name the niftifiles acoording to the stim_file column
    # video_events = events[events['trial_type'] == 'video'].copy()
    video_row = events[(events['trial_type'] == 'video') & 
                       (events['stim_file'].str.contains(CONTENT))].iloc[0]
# %%
# for idx, row in video_events.iterrows():
    # onset = int(np.round(video_row['onset'] / tr))#row['onset']  # in seconds
    # duration = int(np.round(video_row['duration'] / tr))#row['duration']  # in seconds
    #     onset_tr = int(np.round(onset / tr))
    # duration_tr = int(np.round(duration / tr))
    onset_tr = int(np.round(video_row['onset'] / tr))
    duration_tr = int(np.round(video_row['duration'] / tr))
    segment_data = clean_cifti[onset_tr:onset_tr + duration_tr, :]

    stim_file = video_row['stim_file'] if 'stim_file' in video_row else f"video_{idx}"
    base_stimfile = os.path.splitext(os.path.basename(stim_file))[0]
    # Convert time to TRs (volumes)

    end_tr = onset_tr + duration_tr

        # Check bounds
    if onset_tr < 0:
        print(f"Warning: onset_tr {onset_tr} < 0 for {stim_file}")
        onset_tr = 0
    if end_tr > cifti_data.shape[0]:
        print(f"Warning: end_tr {end_tr} > data length {cifti_data.shape[0]} for {stim_file}")
        end_tr = cifti_data.shape[0]



        # --- The Fix: Create a new header for the segmented data ---
    # Get the original axes
    time_axis = dtseries.header.get_axis(0)
    brain_axis = dtseries.header.get_axis(1) 
    
    # Create a new time axis with the new number of timepoints
    new_time_axis = nib.cifti2.SeriesAxis(
        start=0,
        step=time_axis.step,
        size=segment_data.shape[0]
    )
    
    # neuromaps.transforms.fslr_to_fsaverage(segment_data, target_density='41k', hemi=None, method='linear')
    # Create the new CIFTI header
    new_header = nib.cifti2.Cifti2Header.from_axes((new_time_axis, brain_axis))
    new_header = nib.cifti2.Cifti2Header.from_axes((new_time_axis, dtseries.header.get_axis(1)))
    temp_cifti_path = f'/dartfs-hpc/scratch/f0042x1/temp_cleaned_{SUB}_{base_stimfile}.dtseries.nii'
    SCRATCH_DIR = '/dartfs-hpc/scratch/f0042x1'
    nib.save(
        nib.Cifti2Image(
            segment_data,
            header=new_header,
            nifti_header=dtseries.nifti_header),
        temp_cifti_path
    )
    print(f"Video: {stim_file}")
    print(f"  Onset:(TR {onset_tr}) | Duration: ({duration_tr} TRs)")
    print(f"  Extracted shape: {segment_data.shape}")
    
    # Create filename from stim_file
    if 'stim_file' in video_row and pd.notna(video_row['stim_file']):
        # Clean up the stimulus filename for use as output filename
        # base_name = Path(video_row['stim_file']).stem  # Remove extension
        # Replace problematic characters
        safe_name = base_stimfile.replace('/', '_').replace('\\', '_').replace(' ', '_')
        output_filename = f"{safe_name}.dtseries.nii"

    # --- Step 2a: Slice the timeseries using wb_command ---
    
    output_dir = Path('/dartfs/rc/lab/H/HaxbyLab/heejung/data_spacetoptrim')
    sub_output_dir = f'{output_dir}/{SUB}/{SES}/func'
    Path(sub_output_dir).mkdir(parents=True, exist_ok=True)
    temp_segment_path = temp_cifti_path

   
    final_lh_path = f'{SCRATCH_DIR}/{SUB}_{base_stimfile}_lh.func.gii'
    final_rh_path = f'{SCRATCH_DIR}/{SUB}_{base_stimfile}_rh.func.gii'
    
    final_fsaverage_lh_path = f'{sub_output_dir}/{SUB}_{base_stimfile}_lh_fsaverage6.func.gii'
    final_fsaverage_rh_path = f'{sub_output_dir}/{SUB}_{base_stimfile}_rh_fsaverage6.func.gii'
    
    resampled_cifti_path = f'{sub_output_dir}/{SUB}_{base_stimfile}_fsaverage.dtseries.nii'

    NEUROMAP_PATH = '/dartfs/rc/lab/H/HaxbyLab/heejung/neuromaps_template'
    # TODO: point to discovery path
    source_sphere_L = f'{NEUROMAP_PATH}/fsLR/tpl-fsLR_den-32k_hemi-L_sphere.surf.gii'
    target_sphere_L = f'{NEUROMAP_PATH}/fsaverage/tpl-fsaverage_den-41k_hemi-L_sphere.surf.gii'
    source_sphere_R = f'{NEUROMAP_PATH}/fsLR/tpl-fsLR_den-32k_hemi-R_sphere.surf.gii'
    target_sphere_R = f'{NEUROMAP_PATH}/fsaverage/tpl-fsaverage_den-41k_hemi-R_sphere.surf.gii'


    SPLIT_command = [
            'wb_command', '-cifti-separate', str(temp_cifti_path),
            'COLUMN',
            '-metric', 'CORTEX_LEFT', str(final_lh_path),
            '-metric', 'CORTEX_RIGHT', str(final_rh_path)
        ]
    subprocess.run(SPLIT_command, check=True)

    # --------------------
    #  -gifti-convert ASCII 
    # --------------------
    ascii_lh_path = f'{SCRATCH_DIR}/{SUB}_{base_stimfile}_lh_ascii.func.gii'
    ASCII_command = ['wb_command', '-gifti-convert', 'ASCII', str(final_lh_path), str(ascii_lh_path)]


    subprocess.run(ASCII_command, check=True)

    ascii_rh_path = f'{SCRATCH_DIR}/{SUB}_{base_stimfile}_rh_ascii.func.gii'
    ASCII_command = ['wb_command', '-gifti-convert', 'ASCII', str(final_rh_path), str(ascii_rh_path)]
    subprocess.run(ASCII_command, check=True)

    HCP_PATH = '/dartfs/rc/lab/H/HaxbyLab/heejung/neuromaps_template/HCP_fsaverage'
    HCP_fslr2fsaverage_L = f'{HCP_PATH}/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii'
    HCP_fsaverage6std_L = f'{HCP_PATH}/fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii'
    HCP_fslr_mid_L = f'{HCP_PATH}/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii'
    HCP_fsaverage6_mid_L = f'{HCP_PATH}/fsaverage6.L.midthickness_va_avg.41k_fsavg_L.shape.gii'
    # 
    RESAMPLE_command = [
        'wb_command', '-metric-resample' , str(ascii_lh_path),
        HCP_fslr2fsaverage_L,
        HCP_fsaverage6std_L,
        'ADAP_BARY_AREA',
        str(final_fsaverage_lh_path),
        '-area-metrics',
        HCP_fslr_mid_L,
        HCP_fsaverage6_mid_L]
    subprocess.run(RESAMPLE_command, check=True)


    HCP_fslr2fsaverage_R = f'{HCP_PATH}/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii'
    HCP_fsaverage6std_R = f'{HCP_PATH}/fsaverage6_std_sphere.R.41k_fsavg_R.surf.gii'
    HCP_fslr_mid_R = f'{HCP_PATH}/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii'
    HCP_fsaverage6_mid_R = f'{HCP_PATH}/fsaverage6.R.midthickness_va_avg.41k_fsavg_R.shape.gii'
    RESAMPLE_command = [
        'wb_command', '-metric-resample' , str(ascii_rh_path),
        HCP_fslr2fsaverage_R,
        HCP_fsaverage6std_R,
        'ADAP_BARY_AREA',
        str(final_fsaverage_rh_path),
        '-area-metrics',
        HCP_fslr_mid_R,
        HCP_fsaverage6_mid_R]
    subprocess.run(RESAMPLE_command, check=True)
   
    print(f"  {base_stimfile} successfully resampled to fsaverage.")
    
# %% ------------------------------------------------
# Validation: plot derivative fsaverage
# ------------------------------------------------
# import nibabel as nib
# import numpy as np
# from surfplot import Plot
# from nilearn import datasets
# Lfname = '/Users/h/Downloads/ses-01_run-01_order-04_content-parkour_lh_fsaverage6.func.gii'

# Rfname = '/Users/h/Downloads/ses-01_run-01_order-04_content-parkour_rh_fsaverage6.func.gii'
# # 1. Load and average Left Hemisphere
# lh_gii = nib.load(Lfname)
# lh_data = np.column_stack([d.data for d in lh_gii.darrays]).mean(axis=1)

# # 2. Load and average Right Hemisphere
# rh_gii = nib.load(Rfname)
# rh_data = np.column_stack([d.data for d in rh_gii.darrays]).mean(axis=1)

# # 3. Get standard meshes (ensure these match your data's vertex count)
# surfaces = datasets.fetch_surf_fsaverage(mesh='fsaverage6')

# # 4. Plotting
# # We pass both meshes to the Plot constructor
# p = Plot(surf_lh=surfaces['pial_left'], surf_rh=surfaces['pial_right'], size=(1000, 500))

# # 5. Add layers for each hemisphere
# # surfplot handles the data mapping based on which hemisphere is specified
# p.add_layer({'left': lh_data, 'right': rh_data}, cmap='RdBu_r', cbar=True)

# fig = p.build()
# fig.show()