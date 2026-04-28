# %%
import numpy as np
import pandas as pd
import os, re
import scipy.stats
from os.path import join
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from subprocess import call
import nibabel as nib
import matplotlib.pyplot as plt
# from neuromaps.datasets import fetch_fsaverage, fetch_fslr
# from neuromaps.transforms import fsaverage_to_fslr
# from surfplot import Plot

# ── Config ────────────────────────────────────────────────────────────────────
suma_dir     = '/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml'
main_dir     = '/dartfs/rc/lab/H/HaxbyLab/heejung'
result_dir   = '/dartfs/rc/lab/H/HaxbyLab/heejung/data_spacetoptrim'
fsaverage_dir = '/Users/h/neuromaps-data/atlases/fsaverage'
# save_dir_base = '/Users/h/Documents/projects_local/life-encoding/figure/glove_nested'


alignment  = 'ws'
pca_comp   = 40
r_type     = 'r'
clustersize = 50
runs       = [1, 2, 3, 4]
hemis      = ['lh', 'rh']
hemi_dict  = {'lh': 'L', 'rh': 'R'}
n_vertices = 40962
n_medial   = {'lh': 3486, 'rh': 3491}
fdr_thres  = 0.05

features = [
    'scenes-agents-objects-moten',
    'scenes-actions-agents-objects',
    'actions-agents-objects-moten',
    'scenes-actions-agents-moten',
]

subjects = sorted({
    re.search(r'sub-\d+', p.name).group()
    for p in Path(result_dir).glob('sub-*')
    if p.is_dir() and re.search(r'sub-\d+', p.name)
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_dirs(feature):
    return {
        'full':   join(main_dir, 'results', 'revision', 'glove', 'ws_pca-40'),
        'nested': join(main_dir, 'results', 'revision', 'glove_nested', feature, 'ws_pca-40'),
        'output': join(main_dir, 'results', 'revision', 'glove_nested', feature, 'variancepart'),
        'data':   f'{main_dir}/results/revision/glove_nested/{feature}/variancepart',
    }

def get_paths(feature, hemi, dirs):
    d = dirs['data']
    return {
        'mean':        f"{d}/TEMP_hemi-{hemi}_mean.gii",
        'ttest':       f"{d}/variance-{r_type}_{feature}_pca-{pca_comp}_align-{alignment}_hemi-{hemi}_ttest.gii",
        'cluster':     f"{d}/variance-{r_type}_{feature}_pca-{pca_comp}_align-{alignment}_hemi-{hemi}_mean_cluster-{clustersize}.gii",
        'thresholded': f"{d}/variance-{r_type}_{feature}_pca-{pca_comp}_align-{alignment}_hemi-{hemi}_mean_thres-cluster{clustersize}.gii",
        'template':    f"{fsaverage_dir}/tpl-fsaverage_den-41k_hemi-{hemi_dict[hemi]}_inflated.surf.gii",
        'pial':        f"{fsaverage_dir}/tpl-fsaverage_den-41k_hemi-{hemi_dict[hemi]}_pial.surf.gii",
        'pial_suma':   join(suma_dir, f"{hemi}.pial.gii"),
        'medial_mask': join(suma_dir, f'fsaverage6_medial_{hemi}.npy'),
    }

def fisher_mean(data, axis=None):
    return np.tanh(np.nanmean(np.arctanh(data), axis=axis))

def write_gifti(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for _ in range(gii.numDA):
        gii.remove_gifti_data_array(0)
    gii.add_gifti_data_array(nib.gifti.GiftiDataArray(data))
    gii.to_filename(output_fn)

def plot_surface(Lfname, Rfname, label, cmap='inferno', vmin=0, vmax=0.05):
    giiL, giiR = nib.load(Lfname), nib.load(Rfname)
    L_fslr = fsaverage_to_fslr(giiL, target_density='32k', hemi='L', method='linear')
    R_fslr = fsaverage_to_fslr(giiR, target_density='32k', hemi='R', method='linear')
    lh_fslr, rh_fslr = fetch_fslr()['inflated']
    p = Plot(surf_lh=lh_fslr, surf_rh=rh_fslr,
             size=(1000, 200), zoom=1.2, layout='row',
             views=['lateral', 'medial', 'ventral', 'posterior'],
             mirror_views=True, brightness=.7)
    p.add_layer({'left': L_fslr[0], 'right': R_fslr[0]},
                cmap=cmap, cbar=True,
                color_range=(vmin, vmax), cbar_label=label)
    return p.build(cbar_kws=dict(outer_labels_only=True, pad=.02, n_ticks=2, decimals=3))

# ── Step 1: Compute variance partition per feature ────────────────────────────
for feature in features:
    print(f"\n{'='*60}\nProcessing: {feature}\n{'='*60}")
    dirs = get_dirs(feature)
    Path(dirs['output']).mkdir(parents=True, exist_ok=True)

    hemi_mean = []

    for hemi in hemis:
        p = get_paths(feature, hemi, dirs)
        medial_mask     = np.load(p['medial_mask'])
        cortical_coords = np.where(~medial_mask)[0]
        avg_all = []

        for subject in subjects:
            stack_full, stack_nested = [], []
            for run in runs:
                try:
                    full_data   = np.load(f"{dirs['full']}/comb-{r_type}_pca-{pca_comp}_align-{alignment}_{subject}_run-{run}_roi-None_hemi-{hemi}.npy")
                    nested_data = np.load(f"{dirs['nested']}/comb-{r_type}_pca-{pca_comp}_align-{alignment}_{subject}_run-{run}_roi-None_hemi-{hemi}.npy")
                    stack_full.append(full_data)
                    stack_nested.append(nested_data)
                except FileNotFoundError:
                    print(f"  missing: {subject} run-{run}")
                    continue

            if not stack_full:
                continue
            diff    = np.vstack(stack_full) - np.vstack(stack_nested)
            avg_all.append(fisher_mean(diff, axis=0))

        diff_all = np.vstack(avg_all)
        hemi_mean.append(fisher_mean(diff_all, axis=0))

        # Save TEMP mean gifti
        stats = np.zeros(n_vertices)
        stats[cortical_coords] = hemi_mean[-1]
        write_gifti(stats.astype(np.float32),
                    output_fn=p['mean'],
                    template_fn=p['pial_suma'])
        print(f"  [{hemi}] saved mean → {p['mean']}")