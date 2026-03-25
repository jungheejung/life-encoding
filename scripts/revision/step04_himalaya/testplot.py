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
medial_mask = np.load(os.path.join('/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml', f'fsaverage6_medial_{hemisphere}.npy'))
assert np.sum(medial_mask) == n_medial[hemisphere]
cortical_vertices = ~medial_mask # boolean (true for non-medials, false for medials)
cortical_coords = np.where(cortical_vertices)[0] # returns indices of non-medials
         

run_data = np.load('/vast/labs/DBIC/datasets/Life/life-encoding/results/revision/glove/ws_pca-40/comb-r_pca-40_align-ws_sub-0037_run-1_roi-None_hemi-lh.npy')
# fisherz_run = np.arctanh(run_data[0, cortical_vertices])
# stack_fisherz_run.append(fisherz_run)


write_gifti_v2(run_data.astype(np.float32),
template_fn = os.path.join('/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml/fsaverage6_medial_lh.gii'), 
output_fn = '/dartfs/rc/lab/H/HaxbyLab/heejung/TEST_comb-r_pca-40_align-ws_sub-0037_run-1_roi-None_hemi-lh.gii')
