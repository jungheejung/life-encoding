# %%
from neuromaps.datasets import fetch_fsaverage, fetch_fslr
from neuromaps.transforms import fsaverage_to_fslr
from surfplot import Plot
from subprocess import call
from pathlib import Path
import numpy as np
import nibabel as nib
from os.path import join

# ── Config ────────────────────────────────────────────────────────────────────
main_dir     = '/Volumes/results/revision'
fsaverage_dir = '/Users/h/neuromaps-data/atlases/fsaverage'
hemis        = ['lh', 'rh']
hemi_dict    = {'lh': 'L', 'rh': 'R'}
alignment    = 'ws'
clustersize  = 50
r_type       = 'r'

features = [
    'scenes-agents-objects-moten',
    'scenes-actions-agents-objects',
    'actions-agents-objects-moten',
    'scenes-actions-agents-moten', 

    # add more features here
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_data_dir(feature):
    return f'{main_dir}/glove_nested/{feature}/variancepart'

def get_paths(feature, hemi):
    data_dir = get_data_dir(feature)
    return {
        'template':    f"{fsaverage_dir}/tpl-fsaverage_den-41k_hemi-{hemi_dict[hemi]}_inflated.surf.gii",
        'pial':        f"{fsaverage_dir}/tpl-fsaverage_den-41k_hemi-{hemi_dict[hemi]}_pial.surf.gii",
        'mean':        f"{data_dir}/TEMP_hemi-{hemi}_mean.gii",
        'cluster':     f"{data_dir}/variance-{r_type}_{feature}_pca-40_align-{alignment}_hemi-{hemi}_mean_cluster-{clustersize}.gii",
        'thresholded': f"{data_dir}/variance-{r_type}_{feature}_pca-40_align-{alignment}_hemi-{hemi}_mean_thres-cluster{clustersize}.gii",
    }

def fsaverage_to_fslr_and_plot(Lfname, Rfname, key, cmap='inferno', vmin=0, vmax=0.05):
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
                color_range=(vmin, vmax), cbar_label=key)
    return p.build(cbar_kws=dict(outer_labels_only=True, pad=.02, n_ticks=2, decimals=3))

def write_gifti(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for _ in range(gii.numDA):
        gii.remove_gifti_data_array(0)
    gii.add_gifti_data_array(nib.gifti.GiftiDataArray(data))
    gii.to_filename(output_fn)

# ── Step 1: Find clusters (wb_command) ───────────────────────────────────────
for feature in features:
    for hemi in hemis:
        p = get_paths(feature, hemi)
        rc = call(f"wb_command -metric-find-clusters {p['template']} {p['mean']} 0 {clustersize} {p['cluster']}", shell=True)
        print(f"[{feature} | {hemi}] wb_command return code: {rc}")

# ── Step 2: Apply cluster mask ────────────────────────────────────────────────
for feature in features:
    for hemi in hemis:
        p = get_paths(feature, hemi)
        cluster_mask = nib.load(p['cluster']).agg_data() > 0
        mean_data    = nib.load(p['mean']).agg_data()
        write_gifti(
            data=(cluster_mask * mean_data).astype(np.float32),
            output_fn=p['thresholded'],
            template_fn=p['pial']
        )
        print(f"[{feature} | {hemi}] Saved: {p['thresholded']}")

# ── Step 3: Plot ──────────────────────────────────────────────────────────────
for feature in features:
    pL = get_paths(feature, 'lh')['thresholded']
    pR = get_paths(feature, 'rh')['thresholded']
    fig = fsaverage_to_fslr_and_plot(pL, pR, f'{feature}-r', cmap='inferno', vmin=0, vmax=0.01)
    fig.show()
# %%
