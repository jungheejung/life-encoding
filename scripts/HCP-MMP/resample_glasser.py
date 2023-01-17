#
# mri_surf2surf --hemi lh --srcsubject fsaverage --sval ./FILENAME --trgsubject fsaverage6 --tval ./NEWFILENAME
# %%
import nibabel as nib
from nilearn.plotting import plot_surf
from nilearn.plotting import plot_surf_stat_map
import matplotlib.pyplot as plt
from nilearn import datasets
from matplotlib.colors import ListedColormap
fsaverage = datasets.fetch_surf_fsaverage(mesh = 'fsaverage6')

# %%

# labels at each vertex
# colors
# names of labels
lhatlas  = nib.freesurfer.io.read_annot('/Users/h/Dropbox (Dartmouth College)/projects_dropbox/life-encoding/scripts/HCP-MMP/lh.HCP_MMP1.fsaverage6.annot')
lhatlas[0].shape
len(lhatlas[2])
# figure out name of the ROI from glasser atlas
#
# %%
cmap = ListedColormap(lhatlas[1][:,:3]/255)
plot_surf_stat_map(fsaverage['infl_left'], lhatlas[0],
                    bg_map=fsaverage['sulc_left'],title=f'HCP-MMP1.0', 
                    hemi='left', view='lateral', cmap = cmap)  # vmin=.05, vmax=.5,

# %%
