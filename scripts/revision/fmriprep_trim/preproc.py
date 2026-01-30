# load nifti
# use nilearn
# grab fmriprep parameters
# grab beh onset
# nilearn clean image
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

def resample_and_split_hemispheres(cifti_path, output_dir, base_name):
    """
    Resample CIFTI from fsLR to fsaverage using neuromaps and save as separate LH/RH GIFTI files.
    """
    try:
        from neuromaps.transforms import fslr_to_fsaverage
        
        # Load CIFTI data
        cifti_img = nib.load(cifti_path)
        data = cifti_img.get_fdata()
        
        n_left = 29696
        n_right = 29716
        
        left_timepoints = []
        right_timepoints = []
        
        for timepoint in range(data.shape[0]):
            timepoint_data = data[timepoint, :]  # Get timepoint data
            
            # Extract left and right hemisphere data
            left_fsLR = timepoint_data[:n_left]
            right_fsLR = timepoint_data[n_left:n_left+n_right]
            
            # Confirm we're using correct shapes
            print(f"Processing timepoint {timepoint}: left shape: {left_fsLR.shape}, right shape: {right_fsLR.shape}")

            try:
                left_resampled = fslr_to_fsaverage(
                    left_fsLR,
                    target_density='41k',
                    hemi='L',
                    method='linear'
                )
                
                right_resampled = fslr_to_fsaverage(
                    right_fsLR,
                    target_density='41k',
                    hemi='R',
                    method='linear'
                )
                
                left_timepoints.append(left_resampled)
                right_timepoints.append(right_resampled)
                
            except Exception as inner_e:
                print(f"Inner neuromaps error at timepoint {timepoint}: {inner_e}")
                # Fallback to original data if resampling fails
                left_timepoints.append(left_fsLR)
                right_timepoints.append(right_fsLR)

        # Stack and save left and right hemisphere data
        left_data = np.stack(left_timepoints, axis=1)
        right_data = np.stack(right_timepoints, axis=1)

        print(f"Left hemisphere resampled shape: {left_data.shape}")
        print(f"Right hemisphere resampled shape: {right_data.shape}")

        # Save left hemisphere GIFTI
        left_gifti = nib.gifti.GiftiImage()
        for t in range(left_data.shape[1]):
            left_da = nib.gifti.GiftiDataArray(
                left_data[:, t].astype(np.float32).T,
                intent='NIFTI_INTENT_NONE',
                datatype='NIFTI_TYPE_FLOAT32'
            )
            left_gifti.add_gifti_data_array(left_da)

        left_path = Path(output_dir) / f'{base_name}_lh_fsaverage.func.gii'
        nib.save(left_gifti, left_path)
        print(f"Saved left hemisphere: {left_path}")

        # Save right hemisphere GIFTI
        right_gifti = nib.gifti.GiftiImage()
        for t in range(right_data.shape[1]):
            right_da = nib.gifti.GiftiDataArray(
                right_data[:, t].astype(np.float32).T,
                intent='NIFTI_INTENT_NONE',
                datatype='NIFTI_TYPE_FLOAT32'
            )
            right_gifti.add_gifti_data_array(right_da)
        
        right_path = Path(output_dir) / f'{base_name}_rh_fsaverage.func.gii'
        nib.save(right_gifti, right_path)
        print(f"Saved right hemisphere: {right_path}")

        return str(left_path), str(right_path)

    except ImportError:
        print("neuromaps not available. Install with: pip install neuromaps")
        return None, None
    except Exception as e:
        print(f"Error with neuromaps approach: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
# def resample_and_split_hemispheres(cifti_path, output_dir, base_name):
    """
    Resample CIFTI from fsLR to fsaverage using neuromaps and save as separate LH/RH GIFTI files
    """
    try:
        from neuromaps.transforms import fslr_to_fsaverage
        
        # Load CIFTI data
        cifti_img = nib.load(cifti_path)
        data = cifti_img.get_fdata()
        
        print(f"Original data shape: {data.shape}")
        
        # Resample each hemisphere separately using neuromaps
        left_timepoints = []
        right_timepoints = []
        
        for timepoint in range(data.shape[0]):
            timepoint_data = data[timepoint, :]
            
            # Resample left hemisphere only
            left_resampled = fslr_to_fsaverage(
                timepoint_data, 
                target_density='41k',  # fsaverage5 = 10k, fsaverage6 = 41k, fsaverage = 164k
                hemi='L',  # Left hemisphere only
                method='linear'
            )
            
            # Resample right hemisphere only  
            right_resampled = fslr_to_fsaverage(
                timepoint_data,
                target_density='41k',
                hemi='R',  # Right hemisphere only
                method='linear'
            )
            
            left_timepoints.append(left_resampled)
            right_timepoints.append(right_resampled)
            
            if timepoint % 50 == 0:
                print(f"Processed timepoint {timepoint}/{data.shape[0]}")
        
        # Stack timepoints
        left_data = np.stack(left_timepoints, axis=1)  # Shape: (vertices, timepoints)
        right_data = np.stack(right_timepoints, axis=1)
        
        print(f"Left hemisphere resampled shape: {left_data.shape}")
        print(f"Right hemisphere resampled shape: {right_data.shape}")
        
        # Save left hemisphere GIFTI
        left_gifti = nib.gifti.GiftiImage()
        for t in range(left_data.shape[1]):  # Iterate over timepoints
            left_da = nib.gifti.GiftiDataArray(
                left_data[:, t].astype(np.float32),
                intent='NIFTI_INTENT_NONE',
                datatype='NIFTI_TYPE_FLOAT32'
            )
            left_gifti.add_gifti_data_array(left_da)
        
        left_path = Path(output_dir) / f'{base_name}_lh_fsaverage.func.gii'
        nib.save(left_gifti, left_path)
        print(f"Saved left hemisphere: {left_path}")
        
        # Save right hemisphere GIFTI
        right_gifti = nib.gifti.GiftiImage()
        for t in range(right_data.shape[1]):
            right_da = nib.gifti.GiftiDataArray(
                right_data[:, t].astype(np.float32),
                intent='NIFTI_INTENT_NONE',
                datatype='NIFTI_TYPE_FLOAT32'
            )
            right_gifti.add_gifti_data_array(right_da)
            
        right_path = Path(output_dir) / f'{base_name}_rh_fsaverage.func.gii'
        nib.save(right_gifti, right_path)
        print(f"Saved right hemisphere: {right_path}")
        
        return str(left_path), str(right_path)
        
    except ImportError:
        print("neuromaps not available. Install with: pip install neuromaps")
        return None, None
    except Exception as e:
        print(f"Error with neuromaps approach: {e}")
        return None, None
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
    left_path = gifti_dir / f'{base_name}_lh.func.gii'
    nib.save(left_gifti, left_path)
    print(f"  Saved left GIFTI: {left_path}")

    # Save Right Hemisphere as GIFTI
    right_gifti = nib.gifti.GiftiImage()
    
    right_da = nib.gifti.GiftiDataArray(right_data.astype(np.float32), intent='NIFTI_INTENT_NONE')
    right_gifti.add_gifti_data_array(right_da)
    right_path = gifti_dir / f'{base_name}_rh.func.gii'
    nib.save(right_gifti, right_path)
    print(f"  Saved right GIFTI: {right_path}")
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

events[events['trial_type'] == 'video']['onset']
events[events['trial_type'] == 'video']['duration']

# snip out brain data based on onset and duration and 
# name the niftifiles acoording to the stim_file column
video_events = events[events['trial_type'] == 'video'].copy()
# %%
for idx, row in video_events.iterrows():
    onset = row['onset']  # in seconds
    duration = row['duration']  # in seconds
    stim_file = row['stim_file'] if 'stim_file' in row else f"video_{idx}"
    base_stimfile = os.path.splitext(os.path.basename(stim_file))[0]
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
    segment_data = cifti_data[onset_tr:end_tr, :] # (TR, vertex)
    

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
    temp_cifti_path = '/Users/h/Downloads/temp_cleaned.dtseries.nii'
    nib.save(
        nib.Cifti2Image(
            segment_data,
            header=new_header,
            nifti_header=dtseries.nifti_header),
        temp_cifti_path
    )
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

    # --- Step 2a: Slice the timeseries using wb_command ---
    # temp_segment_path = output_filename#output_dir / f"temp_{base_name}.dtseries.nii"
    output_dir = Path('/Users/h/Downloads')
    temp_segment_path = temp_cifti_path
    # output_dir / f"temp_{base_name}.dtseries.nii"
   
    final_lh_path = f'/Users/h/Downloads/{base_stimfile}_lh.func.gii'
    # output_dir / f"{base_name}_lh_fsaverage.func.gii"
    final_rh_path = f'/Users/h/Downloads/{base_stimfile}_rh.func.gii'
    
    final_fsaverage_lh_path = f'/Users/h/Downloads/{base_stimfile}_lh_fsaverage6.func.gii'
    final_fsaverage_rh_path = f'/Users/h/Downloads/{base_stimfile}_rh_fsaverage6.func.gii'
    
    resampled_cifti_path = output_dir / f"{base_name}_fsaverage.dtseries.nii"

    # TODO: point to discovery path
    source_sphere_L = '/Users/h/neuromaps-data/atlases/fsLR/tpl-fsLR_den-32k_hemi-L_sphere.surf.gii'
    target_sphere_L = '/Users/h/neuromaps-data/atlases/fsaverage/tpl-fsaverage_den-41k_hemi-L_sphere.surf.gii'
    source_sphere_R = '/Users/h/neuromaps-data/atlases/fsLR/tpl-fsLR_den-32k_hemi-R_sphere.surf.gii'
    target_sphere_R = '/Users/h/neuromaps-data/atlases/fsaverage/tpl-fsaverage_den-41k_hemi-R_sphere.surf.gii'

    SPLIT_command = [
            'wb_command', '-cifti-separate', str(temp_cifti_path),
            'COLUMN',
            '-metric', 'CORTEX_LEFT', str(final_lh_path),
            '-metric', 'CORTEX_RIGHT', str(final_rh_path)
        ]
    subprocess.run(SPLIT_command, check=True)
    # 

    # 
    #  -gifti-convert ASCII 
    ascii_lh_path = f'/Users/h/Downloads/{base_stimfile}_lh_ascii.func.gii'
    ASCII_command = ['wb_command', '-gifti-convert', 'ASCII', str(final_lh_path), str(ascii_lh_path)]


    subprocess.run(ASCII_command, check=True)

    ascii_rh_path = f'/Users/h/Downloads/{base_stimfile}_rh_ascii.func.gii'
    ASCII_command = ['wb_command', '-gifti-convert', 'ASCII', str(final_rh_path), str(ascii_rh_path)]
    subprocess.run(ASCII_command, check=True)

    # 

    # 
    RESAMPLE_command = [
        'wb_command', '-metric-resample' , str(ascii_lh_path),
        # '/Users/h/neuromaps-data/atlases/fsLR/tpl-fsLR_space-fsaverage_den-32k_hemi-L_sphere.surf.gii',
        '/Users/h/Documents/projects_local/HCP_fsaverage/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii',
        # '/Users/h/neuromaps-data/atlases/fsaverage/tpl-fsaverage_den-41k_hemi-L_sphere.surf.gii',
        '/Users/h/Documents/projects_local/HCP_fsaverage/fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii',
        'ADAP_BARY_AREA',
        str(final_fsaverage_lh_path),
        '-area-metrics',
        '/Users/h/Documents/projects_local/HCP_fsaverage/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii',
        '/Users/h/Documents/projects_local/HCP_fsaverage/fsaverage6.L.midthickness_va_avg.41k_fsavg_L.shape.gii']
    subprocess.run(RESAMPLE_command, check=True)

    RESAMPLE_command = [
        'wb_command', '-metric-resample' , str(ascii_rh_path),
        # '/Users/h/neuromaps-data/atlases/fsLR/tpl-fsLR_space-fsaverage_den-32k_hemi-R_sphere.surf.gii',
        '/Users/h/Documents/projects_local/HCP_fsaverage/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii',
        # '/Users/h/neuromaps-data/atlases/fsaverage/tpl-fsaverage_den-41k_hemi-L_sphere.surf.gii',
        '/Users/h/Documents/projects_local/HCP_fsaverage/fsaverage6_std_sphere.R.41k_fsavg_R.surf.gii',
        'ADAP_BARY_AREA',
        str(final_fsaverage_rh_path),
        '-area-metrics',
        '/Users/h/Documents/projects_local/HCP_fsaverage/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii',
        '/Users/h/Documents/projects_local/HCP_fsaverage/fsaverage6.R.midthickness_va_avg.41k_fsavg_R.shape.gii']
    subprocess.run(RESAMPLE_command, check=True)
   
    print(f"  {base_name} successfully resampled to fsaverage.")
    raise
# %% ------------------------------------------------
# Validation: plot derivative fsaverage
# ------------------------------------------------
import nibabel as nib
import numpy as np
from surfplot import Plot
from nilearn import datasets
Lfname = '/Users/h/Downloads/ses-01_run-01_order-04_content-parkour_lh_fsaverage6.func.gii'

Rfname = '/Users/h/Downloads/ses-01_run-01_order-04_content-parkour_rh_fsaverage6.func.gii'
# 1. Load and average Left Hemisphere
lh_gii = nib.load(Lfname)
lh_data = np.column_stack([d.data for d in lh_gii.darrays]).mean(axis=1)

# 2. Load and average Right Hemisphere
rh_gii = nib.load(Rfname)
rh_data = np.column_stack([d.data for d in rh_gii.darrays]).mean(axis=1)

# 3. Get standard meshes (ensure these match your data's vertex count)
surfaces = datasets.fetch_surf_fsaverage(mesh='fsaverage6')

# 4. Plotting
# We pass both meshes to the Plot constructor
p = Plot(surf_lh=surfaces['pial_left'], surf_rh=surfaces['pial_right'], size=(1000, 500))

# 5. Add layers for each hemisphere
# surfplot handles the data mapping based on which hemisphere is specified
p.add_layer({'left': lh_data, 'right': rh_data}, cmap='RdBu_r', cbar=True)

fig = p.build()
fig.show()