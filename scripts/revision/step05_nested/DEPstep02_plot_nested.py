
# %%
# %%
from neuromaps.datasets import fetch_fsaverage, fetch_fslr
import numpy as np
from surfplot import Plot
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import colors
from os.path import join
import nibabel as nib
surfaces = fetch_fsaverage()
lh, rh = surfaces['inflated']
main_dir = '/Volumes/results/revision' #life-encoding'
# output_dir = os.path.join(main_dir, 'results', 'himalaya', 'glove_nested', 'actions-agents-bg', 'variancepart')
alignment = 'ws'


def fsaverage_to_fslr_and_plot(Lfname, Rfname, key, cmap='inferno', min=0, max=.05):#,min=-40, max=40):
    from neuromaps.datasets import fetch_fslr
    from neuromaps.transforms import fsaverage_to_fslr
    giiL = nib.load(Lfname)
    giiR = nib.load(Rfname)
    L_fslr = fsaverage_to_fslr(giiL, target_density='32k', hemi='L', method='linear')
    R_fslr = fsaverage_to_fslr(giiR, target_density='32k', hemi='R', method='linear')
    
    surfaces_fslr = fetch_fslr()
    lh_fslr, rh_fslr = surfaces_fslr['inflated']
    # color_range = (min,max)
    color_range = (min,max)
    p = Plot(surf_lh=lh_fslr, 
             surf_rh=rh_fslr, 
             size=(1000, 200), 
             zoom=1.2, layout='row', 
             views=['lateral', 'medial', 'ventral', 'posterior'], 
             mirror_views=True, brightness=.7)
    p.add_layer({'left': L_fslr[0], 
                'right': R_fslr[0]}, 
                cmap=cmap, cbar=True,
                color_range=color_range,
                cbar_label=key
                ) # YlOrRd_r
    cbar_kws = dict(outer_labels_only=True, pad=.02, n_ticks=2, decimals=3)
    fig = p.build(cbar_kws=cbar_kws)
    # fig.show()
    return(fig)


from subprocess import call
# %%
# parameteres
hemis = ['lh', 'rh']
# features = ['bg', 'agents', 'actions', 'moten']
clustersize = 50
hemi_dict = {'lh': 'L', 'rh': 'R'}
fsaverage_dir = '/Users/h/neuromaps-data/atlases/fsaverage'

features = ['scenes-agents-objects-moten']
r_type = 'r'
# thresholding via workbench
for hemi in hemis:
    for feature in features:
        data_dir = f'/Volumes/results/revision/glove_nested/{feature}/variancepart'
        fsaverage_template = f"{fsaverage_dir}/tpl-fsaverage_den-41k_hemi-{hemi_dict[hemi]}_inflated.surf.gii"
        # himalaya = f"{data_dir}/variance-{r_type}_{features}_pca-40_align-ha_common_hemi-{hemi}_ttest.gii"
        himalaya = f"{data_dir}/TEMP_hemi-{hemi}_mean.gii"
        outputfname = f"{data_dir}/variance-{r_type}_{feature}_pca-40_align-ws_hemi-{hemi}_mean_cluster-{clustersize}.gii"
        return_code = call(f"wb_command -metric-find-clusters {fsaverage_template} {himalaya} 0 {clustersize} {outputfname}", shell=True)
        
print(f"Return: {return_code}")

# %% [markdown]
# ## check cluster maps
# The images will look like masks

# %%
feature = 'scenes-agents-objects-moten'
clusterthresL = f"{data_dir}/variance-{r_type}_{feature}_pca-40_align-ws_hemi-lh_mean_cluster-{clustersize}.gii"
clusterthresR = f"{data_dir}/variance-{r_type}_{feature}_pca-40_align-ws_hemi-rh_mean_cluster-{clustersize}.gii"
fsaverage_to_fslr_and_plot(clusterthresL, clusterthresR, f'{feature}-r', cmap = 'Reds', min=0, max=.05)


# %%
def write_gifti_v2(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    gda = nib.gifti.GiftiDataArray(data)
    gii.add_gifti_data_array(gda)
    # nib.gifti.write(gii, output_fn)
    gii.to_filename(output_fn)

hemis = ['lh', 'rh']
hemi_dict = {'lh': 'L', 'rh': 'R'}

fsaverage_dir = '/Users/h/neuromaps-data/atlases/fsaverage'

for hemi in hemis:
    for feature in features:
        data_dir = f'/Volumes/results/revision/glove_nested/{feature}/variancepart'
        
        # ✅ changed ttest → mean to match files created above
        clustermask_fname = f"{data_dir}/variance-{r_type}_{feature}_pca-40_align-{alignment}_hemi-{hemi}_mean_cluster-{clustersize}.gii"
        clustL = nib.load(clustermask_fname).agg_data() > 0
        
        # ✅ this is the original mean file (TEMP_hemi-{hemi}_mean.gii source)
        himalaya_fname = f"{data_dir}/TEMP_hemi-{hemi}_mean.gii"
        himalaya = nib.load(himalaya_fname).agg_data()
        
        maskeddata = clustL * himalaya
        write_gifti_v2(maskeddata.astype(np.float32),
            output_fn=f"{data_dir}/variance-{r_type}_{feature}_pca-40_align-{alignment}_hemi-{hemi}_mean_thres-cluster{clustersize}.gii",
            template_fn=f"{fsaverage_dir}/tpl-fsaverage_den-41k_hemi-{hemi_dict[hemi]}_pial.surf.gii")
        # %%
thresL = f"{data_dir}/variance-r_scenes-agents-objects-moten_pca-40_align-ws_hemi-lh_mean_thres-cluster{clustersize}.gii"
thresR = f"{data_dir}/variance-r_scenes-agents-objects-moten_pca-40_align-ws_hemi-rh_mean_thres-cluster{clustersize}.gii"

fig = fsaverage_to_fslr_and_plot(thresL, thresR, 'scenes-agents-objects-moten-r', cmap='inferno', min=0, max=.01)
fig.show()
# # thresholding via workbench
# for hemi in hemis:
#     for feature in features: #features:
#         data_dir = f'/Volumes/results/revision/glove_nested/{feature}/variancepart'
#         clustermask_fname = f"{data_dir}/variance-{r_type}_{feature}_pca-40_align-{alignment}_hemi-{hemi}_ttest_cluster-{clustersize}.gii"
#         clustL = nib.load(clustermask_fname).agg_data() > 0
#         himalaya_fname = f"{data_dir}/variance-{r_type}_{feature}_pca-40_align-{alignment}_hemi-{hemi}_ttest.gii"
#         himalaya = nib.load(himalaya_fname).agg_data()
#         maskeddata = clustL * himalaya
#         write_gifti_v2(maskeddata.astype(np.float32), 
#             output_fn=f"{data_dir}/variance-{r_type}_{feature}_pca-40_align-{alignment}_hemi-{hemi}_ttest_thres-cluster{clustersize}.gii",
#             template_fn=f"{fsaverage_dir}/tpl-fsaverage_den-41k_hemi-{hemi_dict[hemi]}_pial.surf.gii")

# %% [markdown]
# ## plot thresholded maps in fslr [manuscript]

# # %%
# save_dir = f'/Users/h/Documents/projects_local/life-encoding/figure/glove_nested/{features}/variancepart'

# # save_dir = f'/Users/h/jung2heejung@gmail.com - Google Drive/My Drive/life_encoding/figure/glove_nested/variancepart'

# Path(save_dir).mkdir(exist_ok=True, parents=True)
# for feature in nested_list: #['bg', 'agents', 'actions', 'moten']:
#     thresL = f"{data_dir}/variance-{r_type}_{features}_pca-40_align-{alignment}_hemi-lh_ttest_thres-cluster{clustersize}.gii"
#     thresR = f"{data_dir}/variance-{r_type}_{features}_pca-40_align-{alignment}_hemi-rh_ttest_thres-cluster{clustersize}.gii"
#     fig = fsaverage_to_fslr_and_plot(thresL, thresR, f'{features}-r', cmap='inferno',min=0, max=.05)

#     fig.savefig(join(save_dir, f"{features}-r_pca-40_align-{alignment}_hemi-lh_ttest_thres-cluster{clustersize}.png"), dpi=300)

# # %%
# features

# # %%
# join(save_dir, f"{features}-r_pca-40_align-{alignment}_hemi-lh_ttest_thres-cluster{clustersize}10.png")

# # %%
# save_dir = f'/Users/h/Documents/projects_local/life-encoding/figure/glove_nested/{features}/variancepart'

# # save_dir = f'/Users/h/jung2heejung@gmail.com - Google Drive/My Drive/life_encoding/figure/glove_nested/variancepart'

# Path(save_dir).mkdir(exist_ok=True, parents=True)
# for feature in nested_list: #['bg', 'agents', 'actions', 'moten']:
#     thresL = f"{data_dir}/variance-{r_type}_{features}_pca-40_align-{alignment}_hemi-lh_ttest_thres-cluster{clustersize}.gii"
#     thresR = f"{data_dir}/variance-{r_type}_{features}_pca-40_align-{alignment}_hemi-rh_ttest_thres-cluster{clustersize}.gii"
#     fig = fsaverage_to_fslr_and_plot(thresL, thresR, f'{features}-r', cmap='inferno',min=0, max=.10)
#     fig.savefig(join(save_dir, f"{features}-r_pca-40_align-{alignment}_hemi-lh_ttest_thres-cluster{clustersize}10.png"), dpi=300)

# # %%
# print(f"min: {np.nanmin(nib.load(thresL).agg_data())}")
# print(f"max: {np.nanmax(nib.load(thresL).agg_data())}")



# # --------------------------------------------------------------------------------
# # TEMP LOCAL
# # %%--------------------------------------------------------------------------------

# suma_dir = '/Users/h/suma-fsaverage6'

# hemilabels = ['lh', 'rh']
# thresL = f"/Users/h/TEMP_hemi-lh_mean.gii"
# thresR = f"/Users/h/TEMP_hemi-rh_mean.gii"
# stats = np.zeros((n_vertices))
# output_dir = '/Users/h'
# # for h, hemisphere in enumerate(hemis):
# #     hemi_mean = nib.load(f"/Users/h/TEMP_hemi-{hemisphere}_mean.gii").agg_data()
# #     medial_mask = nib.load(f'/Users/h/suma-fsaverage6/{hemisphere}.medial-wall.fsaverage6.gii').agg_data().astype(bool)
# #     # assert np.sum(medial_mask) == n_medial[hemisphere]
# #     cortical_vertices = ~medial_mask # boolean (true for non-medials, false for medials)
# #     cortical_coords = np.where(cortical_vertices)[0] # returns indices of non-medials
# #     stats[cortical_coords] = hemi_mean[h] #[cortical_coords]

# #     save_fname = f"{output_dir}/TMP_hemi_mean_{hemisphere}_wcortical.gii"
# #     write_gifti_v2(stats.astype(np.float32),
# #     template_fn = os.path.join(suma_dir, f"{hemilabels[h]}.pial.gii"), 
# #     output_fn = save_fname)
#      # %%
# plot_dir = '/Users/h/Documents/projects_local/life-encoding/results/revision_plot'
# thresL = f"{plot_dir}/variance-r_scenes-actions-agents-moten_pca-40_align-ws_hemi-lh_ttest.gii"
# thresR = f"{plot_dir}/variance-r_scenes-actions-agents-moten_pca-40_align-ws_hemi-rh_ttest.gii"
# fig = fsaverage_to_fslr_and_plot(thresL, thresR, f'{features}-r', cmap='inferno',min=0, max=.01)
# # %%
# thresL = f"/Users/h/Documents/projects_local/life-encoding/results/revision_plot/TEMP_hemi-lh_mean.gii"
# thresR = f"/Users/h/Documents/projects_local/life-encoding/results/revision_plot/TEMP_hemi-rh_mean.gii"
# # thresL = f"/Users/h/TMP_hemi_mean_lh_wcortical.gii"
# # thresR = f"/Users/h/TMP_hemi_mean_rh_wcortical.gii"
# fig = fsaverage_to_fslr_and_plot(thresL, thresR, f'{features}-r', cmap='inferno',min=0, max=.01)