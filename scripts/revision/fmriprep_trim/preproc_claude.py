"""
Example usage and troubleshooting for CIFTI to fsaverage conversion
"""

# ===== METHOD 1: Using fixed neuromaps approach =====
from pathlib import Path
import neuromaps
import nibabel as nib
# Your CIFTI file path
cifti_path = '/Users/h/Downloads/temp_cleaned.dtseries.nii'
output_dir = '/Users/h/Downloads'
base_name = 'TEST'

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
# Try the fixed neuromaps approach first
left_path, right_path = resample_and_split_hemispheres_fixed(
    cifti_path, 
    output_dir, 
    base_name
)

if left_path and right_path:
    print("Success with neuromaps!")
else:
    print("Neuromaps failed, trying wb_command...")
    
    # ===== METHOD 2: Using wb_command =====
    left_path, right_path = cifti_to_fsaverage_wb_command(
        cifti_path,
        output_dir,
        base_name,
        wb_command_path='wb_command',  # Adjust path if needed
        fsaverage_surfaces_dir='/Users/h/neuromaps-data/atlases'
    )
    
    if left_path and right_path:
        print("Success with wb_command!")
    else:
        print("wb_command failed, saving without resampling...")
        
        # ===== METHOD 3: Direct split without resampling =====
        left_path, right_path = direct_cifti_split_and_save(
            cifti_path,
            output_dir,
            base_name + "_fsLR"  # Add suffix to indicate it's still in fsLR
        )


# ===== TROUBLESHOOTING TIPS =====

"""
Common Issues and Solutions:

1. TRANSPOSE ERROR in original code:
   - Issue: `.T` transpose in GiftiDataArray creation
   - Solution: Remove the `.T` - data should be (vertices,) shape for each timepoint

2. NEUROMAPS IMPORT ERROR:
   - Issue: neuromaps not installed or data not downloaded
   - Solution: 
     pip install neuromaps
     # Then in Python:
     from neuromaps.datasets import fetch_atlases
     fetch_atlases(atlas=['fsLR', 'fsaverage'])

3. WB_COMMAND NOT FOUND:
   - Issue: Connectome Workbench not installed or not in PATH
   - Solution:
     # Mac: brew install connectome-workbench
     # Linux: download from humanconnectome.org
     # Then specify full path in function call

4. SPHERE FILES NOT FOUND:
   - Issue: Surface sphere files needed for resampling not available
   - Solution: Download from neuromaps or HCP:
     from neuromaps.datasets import fetch_fslr, fetch_fsaverage
     surfaces_fslr = fetch_fslr()
     surfaces_fsaverage = fetch_fsaverage()

5. MEMORY ERROR with large files:
   - Issue: Loading entire timeseries at once
   - Solution: Process in chunks:
"""

def process_cifti_in_chunks(cifti_path, output_dir, base_name, chunk_size=100):
    """Process large CIFTI files in chunks to avoid memory issues"""
    from neuromaps.transforms import fslr_to_fsaverage
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    
    cifti_img = nib.load(cifti_path)
    n_timepoints = cifti_img.shape[0]
    n_left = 29696
    n_right = 29716
    
    # Initialize output GIFTI files
    left_gifti = nib.gifti.GiftiImage()
    right_gifti = nib.gifti.GiftiImage()
    
    for start_idx in range(0, n_timepoints, chunk_size):
        end_idx = min(start_idx + chunk_size, n_timepoints)
        print(f"Processing timepoints {start_idx} to {end_idx}")
        
        # Load only this chunk
        chunk_data = cifti_img.dataobj[start_idx:end_idx, :]
        
        for t_idx, t_global in enumerate(range(start_idx, end_idx)):
            timepoint_data = chunk_data[t_idx, :]
            
            # Extract hemispheres
            left_fsLR = timepoint_data[:n_left]
            right_fsLR = timepoint_data[n_left:n_left+n_right]
            
            # Transform
            left_resampled = fslr_to_fsaverage(left_fsLR, '41k', 'L', 'linear')
            right_resampled = fslr_to_fsaverage(right_fsLR, '41k', 'R', 'linear')
            
            # Add to GIFTI
            left_da = nib.gifti.GiftiDataArray(
                left_resampled.astype(np.float32),
                intent='NIFTI_INTENT_NONE'
            )
            right_da = nib.gifti.GiftiDataArray(
                right_resampled.astype(np.float32),
                intent='NIFTI_INTENT_NONE'
            )
            
            left_gifti.add_gifti_data_array(left_da)
            right_gifti.add_gifti_data_array(right_da)
    
    # Save files
    output_dir = Path(output_dir)
    left_path = output_dir / f'{base_name}_lh_fsaverage.func.gii'
    right_path = output_dir / f'{base_name}_rh_fsaverage.func.gii'
    
    nib.save(left_gifti, left_path)
    nib.save(right_gifti, right_path)
    
    return str(left_path), str(right_path)


# ===== VALIDATION =====

def validate_gifti_output(gifti_path):
    """Validate that GIFTI file is correctly formatted"""
    import nibabel as nib
    
    try:
        gii = nib.load(gifti_path)
        n_timepoints = len(gii.darrays)
        n_vertices = gii.darrays[0].data.shape[0] if gii.darrays else 0
        
        print(f"GIFTI validation for {gifti_path}:")
        print(f"  - Number of timepoints: {n_timepoints}")
        print(f"  - Number of vertices: {n_vertices}")
        print(f"  - Data type: {gii.darrays[0].datatype if gii.darrays else 'N/A'}")
        print(f"  - Intent: {gii.darrays[0].intent if gii.darrays else 'N/A'}")
        
        # Check for fsaverage6 (41k vertices)
        if n_vertices == 40962:
            print("  ✓ Matches fsaverage6 (41k) vertex count")
        elif n_vertices == 163842:
            print("  ✓ Matches fsaverage (164k) vertex count")
        elif n_vertices == 29696 or n_vertices == 29716:
            print("  ⚠ Still in fsLR space (32k)")
        else:
            print(f"  ⚠ Unexpected vertex count")
            
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

# Validate your outputs
if left_path and right_path:
    validate_gifti_output(left_path)
    validate_gifti_output(right_path)