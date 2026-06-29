import numpy as np
import pandas as pd
import os
from os.path import join
import scipy.stats
from statsmodels.stats.multitest import multipletests
import nibabel as nib
from pathlib import Path

from nilearn import datasets
from nilearn.image import smooth_img
# from nilearn.datasets import (
#     # load_fsaverage,
#     load_fsaverage_data
# )
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import SurfaceImage

def write_gifti_v2(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    gda = nib.gifti.GiftiDataArray(data)
    gii.add_gifti_data_array(gda)
    gii.to_filename(output_fn)

def bootstrap_test(data, n_resample=1000, estimator=np.mean, axis=0):
    n_subjects = data.shape[axis]
    true_stat = estimator(data, axis=axis)
    boot_dist = []
    for i in np.arange(n_resample):
        boot_ids = np.random.choice(np.arange(n_subjects), size=n_subjects, replace=True)
        boot_sample = data[boot_ids]
        boot_stat = estimator(boot_sample, axis=axis)
        boot_dist.append(boot_stat)
    boot_dist = np.array(boot_dist)
    boot_null = boot_dist - true_stat
    p = (np.sum(boot_null >= true_stat, axis=axis) + 1) / (n_resample + 1)
    return p

def fisher_mean(correlation, axis=None):
    return np.tanh(np.nanmean(np.arctanh(correlation), axis=axis))

# =============================================================================
# Parameters
# =============================================================================
suma_dir  = '/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml'
main_dir  = '/dartfs/rc/lab/H/HaxbyLab/heejung'
alignment = 'ws'
pca_comp  = 40
r_type    = 'r'
runs      = [1, 2, 3, 4]
hemis     = ['lh', 'rh']
hemi_dict = {'lh': 'left', 'rh': 'right'}
n_vertices = 40962
n_medial   = {'lh': 3486, 'rh': 3491}
fdr_thres  = 0.05
multiplecomp = 'fdr' #'fdr'

subjects = ['sub-0002', 'sub-0008', 'sub-0010', 'sub-0011', 'sub-0013', 'sub-0014', 'sub-0016', 'sub-0018', 'sub-0019', 'sub-0020', 'sub-0026', 'sub-0031', 'sub-0032', 'sub-0033', 'sub-0034', 'sub-0036', 'sub-0037', 'sub-0038', 'sub-0040', 'sub-0043', 'sub-0044', 'sub-0046', 'sub-0050', 'sub-0051', 'sub-0052', 'sub-0053', 'sub-0055', 'sub-0058', 'sub-0059', 'sub-0060', 'sub-0062', 'sub-0064', 'sub-0065', 'sub-0066', 'sub-0069', 'sub-0070', 'sub-0073', 'sub-0075', 'sub-0076', 'sub-0077', 'sub-0078', 'sub-0079', 'sub-0080', 'sub-0084', 'sub-0086', 'sub-0087', 'sub-0088', 'sub-0089', 'sub-0090', 'sub-0092', 'sub-0093', 'sub-0095', 'sub-0099', 'sub-0101', 'sub-0102', 'sub-0104', 'sub-0106', 'sub-0107', 'sub-0109', 'sub-0111', 'sub-0112', 'sub-0115', 'sub-0116', 'sub-0122', 'sub-0124', 'sub-0126', 'sub-0127', 'sub-0129', 'sub-0130', 'sub-0131', 'sub-0132', 'sub-0133']

# each folder = full model MINUS one feature → that missing feature is the unique variance
folders = {
    'actions-agents-objects-moten': 'scenes',    # missing scenes
    'scenes-actions-agents-objects': 'moten',    # missing moten
    'scenes-agents-objects-moten':   'actions',  # missing actions
    'scenes-actions-agents-moten':   'objects',  # missing objects
    'scenes-actions-objects-moten':  'agents',   # missing agents
}

full_dir = os.path.join(main_dir, 'results', 'revision', 'glove', f'ws_pca-{pca_comp}')

# =============================================================================
# Main loop over nested folders
# =============================================================================
# diff_allfeatures = []
diff_allfeatures = {}
p_all_nans =[]
arctan_nans = []
for folder, unique_feature in folders.items():
    print(f"\n{'='*60}")
    print(f"Processing: {folder}  →  unique variance of: {unique_feature}")
    print(f"{'='*60}")

    nested_dir = os.path.join(main_dir, 'results', 'revision', 'glove_nested', folder, f'ws_pca-{pca_comp}')
    output_dir = os.path.join(main_dir, 'results', 'revision', 'glove_nested', folder, 'variancepart')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    hemi_p    = []
    hemi_mean = []
    fsaverage6 = datasets.fetch_surf_fsaverage('fsaverage6')
    diff_allfeatures.setdefault(unique_feature, {})

    for hemisphere in hemis:
        medial_mask     = np.load(join(suma_dir, f'fsaverage6_medial_{hemisphere}.npy'))
        assert np.sum(medial_mask) == n_medial[hemisphere]
        cortical_vertices = ~medial_mask
        cortical_coords   = np.where(cortical_vertices)[0]

        avg_all = []

        for test_subject in subjects:
            stack_fullrun   = []
            stack_nestedrun = []

            for test_run in runs:
                full_fname   = f"{full_dir}/comb-{r_type}_pca-{pca_comp}_align-{alignment}_{test_subject}_run-{test_run}_roi-None_hemi-{hemisphere}.npy"
                nested_fname = f"{nested_dir}/comb-{r_type}_pca-{pca_comp}_align-{alignment}_{test_subject}_run-{test_run}_roi-None_hemi-{hemisphere}.npy"

                try:
                    full_data   = np.load(full_fname)
                    nested_data = np.load(nested_fname)
                except FileNotFoundError as e:
                    print(f"  missing: {e}")
                    continue

                stack_fullrun.append(full_data)
                stack_nestedrun.append(nested_data)

            if len(stack_fullrun) == 0:
                print(f"  skipping {test_subject}: no valid runs")
                continue

            stack_full   = np.vstack(stack_fullrun)
            stack_nested = np.vstack(stack_nestedrun)
            diff         = stack_full - stack_nested
            avg_run      = fisher_mean(diff, axis=0)
            # TODO: -----------------------------
            # Smooth data. npy -> .gii -> smooth via nilearn -> npy
            # Full code
            # 4. Create the SurfaceImage surf_image = SurfaceImage( mesh={ "left": fsaverage6["pial_left"], "right": fsaverage6["pial_right"] }, data={ "left": left_data, "right": right_data } )

            avg_run_wmedial = np.zeros(n_vertices)
            avg_run_wmedial[cortical_coords] = avg_run
            npy2surf = SurfaceImage( mesh={ 
                hemi_dict[hemisphere]: fsaverage6[f"pial_{hemi_dict[hemisphere]}"]}, 
                data={ hemi_dict[hemisphere]: avg_run_wmedial } )

            avg_run_smooth = smooth_img(npy2surf, fwhm=10)
            avg_run_smooth_arr = avg_run_smooth.data.parts[hemi_dict[hemisphere]]
            avg_run_smooth_arr_wo_medial = avg_run_smooth_arr[cortical_vertices]
            # -----------------------------------
            avg_all.append(avg_run_smooth_arr_wo_medial)

        if len(avg_all) == 0:
            print(f"  WARNING: no subjects for {hemisphere}, skipping")
            continue

        diff_all = np.vstack(avg_all)  # n_subjects x n_vertices
    # diff_allfeatures[unique_feature] = diff_all
        diff_allfeatures.setdefault(unique_feature, {})[hemisphere] = diff_all

    print(f"  done: {unique_feature}")


# per subject, find argmax feature and binarize per vertex

# diff_allfeatures['scenes'].shape[0]

# # stack diff all across features/subjects
# per_sub_stack = []
# group_stack = []
# for sub in np.arange(diff_allfeatures['scenes'].shape[0]):
#     within_sub_stack = []
#     for feature in diff_allfeatures.keys():
#         within_sub_stack.append(diff_allfeatures[feature][sub])
#     per_sub_stack = np.vstack(within_sub_stack)
#     group_stack.append(per_sub_stack)    
# =============================================================================
# Concatenate lh + rh per subject, per feature, into one whole-brain array
# =============================================================================
diff_allfeatures_combined = {}
for feature in diff_allfeatures:
    lh_data = diff_allfeatures[feature]['lh']  # n_subjects x n_vertices_lh
    rh_data = diff_allfeatures[feature]['rh']  # n_subjects x n_vertices_rh
    assert lh_data.shape[0] == rh_data.shape[0], "subject count mismatch between hemispheres"
    diff_allfeatures_combined[feature] = np.concatenate([lh_data, rh_data], axis=1)  # n_subjects x (n_vertices_lh + n_vertices_rh)

n_subjects_final = diff_allfeatures_combined['scenes'].shape[0]

# =============================================================================
# Per subject, find argmax feature and binarize per vertex (now whole-brain)
# =============================================================================
group_stack = []
for sub in np.arange(n_subjects_final):
    within_sub_stack = [diff_allfeatures_combined[feature][sub] for feature in diff_allfeatures_combined]
    per_sub_stack = np.vstack(within_sub_stack)  # n_features x n_vertices_total
    group_stack.append(per_sub_stack)

# get argmax for features
argmax_list = []
for sub in np.arange(len(group_stack)):
    argmax_list.append(np.argmax(group_stack[sub], axis=0))

argmax_np = np.vstack(argmax_list)


# count feature vertices
counts = {}
for sub in range(argmax_np.shape[0]):
    row_counts = np.bincount(argmax_np[sub], minlength=len(diff_allfeatures_combined))
    for i, key in enumerate(diff_allfeatures_combined.keys()):
        counts.setdefault(key, []).append(row_counts[i])

counts = {key:np.array(v) for key, v in counts.items()}

