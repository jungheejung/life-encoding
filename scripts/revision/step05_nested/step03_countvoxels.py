# %%
import nibabel as nib
import numpy as np
mean_dir = '/Volumes/results/revision/glove_nested'
feature = 'actions'
feature_dict = {
    'scenes': 'actions-agents-objects-moten',
    'objects':'scenes-actions-agents-moten',
    'moten': 'scenes-actions-agents-objects',
    'agents': 'scenes-actions-objects-moten',
    'actions': 'scenes-agents-objects-moten'
    }




# %%
# feature_list = list(feature_dict.keys())
n_vertices = 40962
n_medial = {'lh': 3486, 'rh': 3491}
hemi_dict = {'lh':{}, 'rh': {}}
for hemisphere in ['lh', 'rh']:
    # load data 
    # _______________________________________________________
    vertex_minus_medial = n_vertices - n_medial[hemisphere]
    scene_string = feature_dict['scenes']
    scenes_gii = np.mean(nib.load(f'{mean_dir}/{scene_string}/variancepart/variance-r_scenes_pca-40_align-ws_hemi-{hemisphere}_multcomp-unc_mean.gii').agg_data(), axis=0)

    object_string = feature_dict['objects']
    objects_gii = np.mean(nib.load(f'{mean_dir}/{object_string}/variancepart/variance-r_objects_pca-40_align-ws_hemi-{hemisphere}_multcomp-unc_mean.gii').agg_data(), axis=0)

    moten_string = feature_dict['moten']
    moten_gii = np.mean(nib.load(f'{mean_dir}/{moten_string}/variancepart/variance-r_moten_pca-40_align-ws_hemi-{hemisphere}_multcomp-unc_mean.gii').agg_data(), axis=0)

    agent_string = feature_dict['agents']
    agents_gii = np.mean(nib.load(f'{mean_dir}/{agent_string}/variancepart/variance-r_agents_pca-40_align-ws_hemi-{hemisphere}_multcomp-unc_mean.gii').agg_data(), axis=0)

    action_string = feature_dict['actions']
    actions_gii = np.mean(nib.load(f'{mean_dir}/{action_string}/variancepart/variance-r_actions_pca-40_align-ws_hemi-{hemisphere}_multcomp-unc_mean.gii').agg_data(), axis=0)
    

    # find voxels greater than other features ______________________________
    hemi_dict[hemisphere]['actions'] = np.sum(
        (actions_gii > objects_gii) & 
        (actions_gii > scenes_gii) &
        (actions_gii > agents_gii) &
        (actions_gii > moten_gii)
        ) / vertex_minus_medial

    hemi_dict[hemisphere]['scenes'] = np.sum(
        (scenes_gii > objects_gii) & 
        (scenes_gii > actions_gii) &
        (scenes_gii > agents_gii) &
        (scenes_gii > moten_gii)
        ) / vertex_minus_medial

    hemi_dict[hemisphere]['objects'] = np.sum(
        (objects_gii > actions_gii) & 
        (objects_gii > scenes_gii) &
        (objects_gii > agents_gii) &
        (objects_gii > moten_gii)
        ) / vertex_minus_medial
    
    hemi_dict[hemisphere]['moten'] = np.sum(
        (moten_gii > objects_gii) & 
        (moten_gii > scenes_gii) &
        (moten_gii > agents_gii) &
        (moten_gii > actions_gii)
        ) / vertex_minus_medial

    hemi_dict[hemisphere]['agents'] = np.sum(
        (agents_gii > objects_gii) & 
        (agents_gii > scenes_gii) &
        (agents_gii > moten_gii) &
        (agents_gii > actions_gii)
        ) / vertex_minus_medial


# %%

for feature in feature_dict.keys():
    print(f"{feature}")
    print(np.mean([hemi_dict['lh'][feature], hemi_dict['rh'][feature]]))
# %%
