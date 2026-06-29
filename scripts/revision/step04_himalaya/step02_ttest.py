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
from nilearn.surface import SurfaceImage


def write_gifti_v2(data, output_fn, template_fn):
    """Write a 1-D float32 array to a gifti file using template metadata."""
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    gda = nib.gifti.GiftiDataArray(data)
    gii.add_gifti_data_array(gda)
    gii.to_filename(output_fn)


def fisher_mean(correlation, axis=None):
    return np.tanh(np.nanmean(np.arctanh(correlation), axis=axis))


# =============================================================================
# Parameters
# =============================================================================
suma_dir     = '/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml'
main_dir     = '/dartfs/rc/lab/H/HaxbyLab/heejung'
alignment    = 'ws'
pca_comp     = 40
r_type       = 'r'
runs         = [1, 2, 3, 4]
hemis        = ['lh', 'rh']
hemi_dict    = {'lh': 'left', 'rh': 'right'}
n_vertices   = 40962
n_medial     = {'lh': 3486, 'rh': 3491}
fdr_thres    = 0.05
multiplecomp = 'unc'

subjects = [
    'sub-0002', 'sub-0008', 'sub-0010', 'sub-0011', 'sub-0013', 'sub-0014',
    'sub-0016', 'sub-0018', 'sub-0019', 'sub-0020', 'sub-0026', 'sub-0031',
    'sub-0032', 'sub-0033', 'sub-0034', 'sub-0036', 'sub-0037', 'sub-0038',
    'sub-0040', 'sub-0043', 'sub-0044', 'sub-0046', 'sub-0050', 'sub-0051',
    'sub-0052', 'sub-0053', 'sub-0055', 'sub-0058', 'sub-0059', 'sub-0060',
    'sub-0062', 'sub-0064', 'sub-0065', 'sub-0066', 'sub-0069', 'sub-0070',
    'sub-0073', 'sub-0075', 'sub-0076', 'sub-0077', 'sub-0078', 'sub-0079',
    'sub-0080', 'sub-0084', 'sub-0086', 'sub-0087', 'sub-0088', 'sub-0089',
    'sub-0090', 'sub-0092', 'sub-0093', 'sub-0095', 'sub-0099', 'sub-0101',
    'sub-0102', 'sub-0104', 'sub-0106', 'sub-0107', 'sub-0109', 'sub-0111',
    'sub-0112', 'sub-0115', 'sub-0116', 'sub-0122', 'sub-0124', 'sub-0126',
    'sub-0127', 'sub-0129', 'sub-0130', 'sub-0131', 'sub-0132', 'sub-0133',
]

features = ['objects','moten', 'scenes', 'actions', 'agents']

full_dir   = os.path.join(main_dir, 'results', 'revision', 'glove', f'ws_pca-{pca_comp}')
output_dir = os.path.join(main_dir, 'results', 'revision', 'glove', 'fullmodel_ttest')
Path(output_dir).mkdir(parents=True, exist_ok=True)

fsaverage6 = datasets.fetch_surf_fsaverage('fsaverage6')

# =============================================================================
# Main loop: feature → hemisphere → subject → run
# =============================================================================
for feature in features:
    print(f"\n{'='*60}")
    print(f"Processing feature: {feature}")
    print(f"{'='*60}")

    # FIX 1: reset accumulators per feature, not once globally
    hemi_p    = []
    hemi_mean = []

    for hemisphere in hemis:
        medial_mask       = np.load(join(suma_dir, f'fsaverage6_medial_{hemisphere}.npy'))
        assert np.sum(medial_mask) == n_medial[hemisphere]
        cortical_vertices = ~medial_mask
        cortical_coords   = np.where(cortical_vertices)[0]

        avg_all = []   # (n_subjects, n_cortical_vertices) after smoothing

        for test_subject in subjects:
            stack_runs = []

            for test_run in runs:
                full_fname = (
                    f"{full_dir}/{feature}-r/{feature}-{r_type}_pca-{pca_comp}_align-{alignment}"
                    f"_{test_subject}_run-{test_run}_hemi-{hemisphere}.npy"
                )
                try:
                    full_data = np.load(full_fname)
                except FileNotFoundError as e:
                    print(f"  missing: {e}")
                    continue
                except ValueError as e:
                    print(f"  bad array (object dtype?) in {full_fname}: {e}")
                    continue

                if full_data.dtype == object:
                    print(f"  object dtype in {full_fname}, skipping")
                    continue
                full_data = full_data.squeeze()
                if full_data.ndim != 1:
                    print(f"  unexpected shape {full_data.shape} after squeeze in {full_fname}, skipping")
                    continue

                stack_runs.append(full_data)

            if len(stack_runs) == 0:
                print(f"  skipping {test_subject}: no valid runs")
                continue

            # Average across runs in Fisher-z space
            stack_runs_arr = np.vstack(stack_runs)                # (n_runs, n_cortical_vertices)
            avg_run        = fisher_mean(stack_runs_arr, axis=0)  # (n_cortical_vertices,)

            # ------------------------------------------------------------------
            # Smooth: embed cortical values into full surface, smooth, strip back
            # ------------------------------------------------------------------
            avg_run_wmedial = np.zeros(n_vertices)
            avg_run_wmedial[cortical_coords] = avg_run

            surf_img = SurfaceImage(
                mesh={hemi_dict[hemisphere]: fsaverage6[f"pial_{hemi_dict[hemisphere]}"]},
                data={hemi_dict[hemisphere]: avg_run_wmedial},
            )
            smoothed_img      = smooth_img(surf_img, fwhm=10)
            smoothed_arr      = smoothed_img.data.parts[hemi_dict[hemisphere]]
            smoothed_cortical = smoothed_arr[cortical_vertices]   # (n_cortical_vertices,)

            avg_all.append(smoothed_cortical)

        if len(avg_all) == 0:
            print(f"  WARNING: no subjects for {hemisphere}, skipping")
            continue

        data_all = np.vstack(avg_all)   # (n_subjects, n_cortical_vertices)

        # FIX 2+3: save the cross-subject mean (1-D, full surface) with feature in filename
        mean_cortical = fisher_mean(data_all, axis=0)   # (n_cortical_vertices,)
        mean_wmedial  = np.zeros(n_vertices, dtype=np.float32)
        mean_wmedial[cortical_coords] = mean_cortical

        mean_save_fname = (
            f"{output_dir}/{feature}-{r_type}_pca-{pca_comp}_align-{alignment}"
            f"_hemi-{hemisphere}_mean.gii"
        )
        write_gifti_v2(
            mean_wmedial,
            template_fn=join(suma_dir, f'{hemisphere}.pial.gii'),
            output_fn=mean_save_fname,
        )

        # One-sample t-test (greater than 0) in Fisher-z space
        t, p = scipy.stats.ttest_1samp(
            np.arctanh(data_all), popmean=0, axis=0, alternative='greater'
        )

        # Replace NaNs (zero-variance vertices) with p=1 to preserve shape
        pvals_replaced = np.where(np.isnan(p), 1.0, p)

        hemi_p.append(pvals_replaced)
        hemi_mean.append(mean_cortical)

    # -------------------------------------------------------------------------
    # FDR correction pooled across both hemispheres
    # -------------------------------------------------------------------------
    if len(hemi_p) < 2:
        print(f"  WARNING: fewer than 2 hemispheres completed for {feature}, skipping FDR")
        continue

    left_vert = hemi_p[0].shape[0]
    p_all     = np.hstack(hemi_p)

    reject, q_all, _, _ = multipletests(p_all, method='fdr_bh')
    q_both = [q_all[:left_vert], q_all[left_vert:]]
    p_both = [p_all[:left_vert], p_all[left_vert:]]
    print("  completed t-test + FDR correction")

    # -------------------------------------------------------------------------
    # Save thresholded gifti per hemisphere
    # -------------------------------------------------------------------------
    for h, hemisphere in enumerate(hemis):
        medial_mask       = np.load(join(suma_dir, f'fsaverage6_medial_{hemisphere}.npy'))
        cortical_vertices = ~medial_mask
        cortical_coords   = np.where(cortical_vertices)[0]

        mean_h = hemi_mean[h].copy()

        if multiplecomp == 'fdr':
            mean_h[q_both[h] >= fdr_thres] = 0
        elif multiplecomp == 'unc':
            mean_h[p_both[h] >= fdr_thres] = 0

        stats = np.zeros(n_vertices, dtype=np.float32)
        stats[cortical_coords] = mean_h

        save_fname = (
            f"{output_dir}/{feature}-{r_type}_pca-{pca_comp}_align-{alignment}"
            f"_hemi-{hemisphere}_multcomp-{multiplecomp}_ttest.gii"
        )
        write_gifti_v2(
            stats,
            template_fn=join(suma_dir, f'{hemisphere}.pial.gii'),
            output_fn=save_fname,
        )
        print(f"  saved: {save_fname}")

    print(f"  done: {feature}")