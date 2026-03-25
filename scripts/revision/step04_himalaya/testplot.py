# for result in results:
    # hemi_t = []
    # hemi_p = []
    # hemi_mean = []
    # output_dir = os.path.join(results_dir,f"{result.strip('-r')}", 'ha_common_pca-40')
    # print(f"starting {result} ________________")
    # for hemisphere in hemis:
import numpy as np
import pandas as pd
import os
from os.path import join
import scipy.stats
# from statsmodels.stats.multitest import multipletests
import nibabel as nib
from pathlib import Path

def write_gifti_v2(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    gda = nib.gifti.GiftiDataArray(data)
    gii.add_gifti_data_array(gda)
    # nib.gifti.write(gii, output_fn)
    gii.to_filename(output_fn)

n_medial = {'lh': 3486, 'rh': 3491}
hemisphere = 'lh'
n_vertices = 40962
medial_mask = np.load(os.path.join('/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml', f'fsaverage6_medial_{hemisphere}.npy'))
assert np.sum(medial_mask) == n_medial[hemisphere]
cortical_vertices = ~medial_mask # boolean (true for non-medials, false for medials)
cortical_coords = np.where(cortical_vertices)[0] # returns indices of non-medials
         

run_data = np.load('/vast/labs/DBIC/datasets/Life/life-encoding/results/revision/glove/ws_pca-40/comb-r_pca-40_align-ws_sub-0037_run-1_roi-None_hemi-lh.npy')
stats = np.zeros((n_vertices))
# fisherz_run = np.arctanh(run_data[0, cortical_vertices])
# stack_fisherz_run.append(fisherz_run)
stats[cortical_coords] = run_data

write_gifti_v2(stats.astype(np.float32),
template_fn = os.path.join('/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml/fsaverage6_medial_lh.gii'), 
output_fn = '/dartfs/rc/lab/H/HaxbyLab/heejung/TEST_comb-r_pca-40_align-ws_sub-0037_run-1_roi-None_hemi-lh.gii')


# %%
# from neuromaps.datasets import fetch_fsaverage, fetch_fslr
# import numpy as np
# from surfplot import Plot
# import nibabel as nib
# import matplotlib.pyplot as plt
# from matplotlib import colors
# from os.path import join
# import nibabel as nib
# def fsaverage_to_fslr_and_plot(Lfname, Rfname, key, cmap='inferno', max=.20):
#     from neuromaps.datasets import fetch_fslr
#     from neuromaps.transforms import fsaverage_to_fslr
#     giiL = nib.load(Lfname)
#     giiR = nib.load(Rfname)
#     L_fslr = fsaverage_to_fslr(giiL, target_density='32k', hemi='L', method='linear')
#     R_fslr = fsaverage_to_fslr(giiR, target_density='32k', hemi='R', method='linear')
    
#     surfaces_fslr = fetch_fslr()
#     lh_fslr, rh_fslr = surfaces_fslr['inflated']

#     color_range = (0,max)
#     p = Plot(surf_lh=lh_fslr, 
#              surf_rh=rh_fslr, 
#              size=(1000, 200), 
#              zoom=1.2, layout='row', 
#              views=['lateral', 'medial', 'ventral', 'posterior'], 
#              mirror_views=True, brightness=.7)
#     p.add_layer({'left': L_fslr[0], 
#                 'right': R_fslr[0]}, 
#                 cmap=cmap, cbar=True,
#                 color_range=color_range,
#                 cbar_label=key
#                 ) # YlOrRd_r
#     cbar_kws = dict(outer_labels_only=True, pad=.02, n_ticks=2, decimals=3)
#     fig = p.build(cbar_kws=cbar_kws)
#     # fig.show()
#     return(fig)

# # %%
# Lfname = '/Users/h/Documents/projects_local/life-encoding/TEST_comb-r_pca-40_align-ws_sub-0037_run-1_roi-None_hemi-lh.gii'
# fsaverage_to_fslr_and_plot(Lfname, Lfname, key='test', cmap='inferno', max=.20)
# %%
