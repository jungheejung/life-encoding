# %%
import nibabel as nib
import numpy as np
mean_dir = '/Volumes/results/revision/glove_nested'
feature = 'actions'
feature_dict = {
    'scenes': 'actions-agents-objects-moten',
    'objects':'scenes-actions-agents-moten',
    'moten': 'scenes-actions-agents-objects',
    # 'agents': 'scenes-actions-objects-moten',
    'actions': 'scenes-agents-objects-moten'
    }




# %%
# feature_list = list(feature_dict.keys())
hemi_dict = {'lh':{}, 'rh': {}}
for hemisphere in ['lh', 'rh']:
    # load data _______________________________________________________
    scene_string = feature_dict['scenes']
    scenes_gii = nib.load(f'{mean_dir}/{scene_string}/variancepart/TEMP_hemi-{hemisphere}_mean.gii').agg_data()

    object_string = feature_dict['objects']
    objects_gii = nib.load(f'{mean_dir}/{object_string}/variancepart/TEMP_hemi-{hemisphere}_mean.gii').agg_data()

    moten_string = feature_dict['moten']
    moten_gii = nib.load(f'{mean_dir}/{moten_string}/variancepart/TEMP_hemi-{hemisphere}_mean.gii').agg_data()

    # agents_gii = nib.load(f'{mean_dir}/{feature_dict['agents']}/variancepart/TEMP_hemi-{hemisphere}_mean.gii')

    action_string = feature_dict['actions']
    actions_gii = nib.load(f'{mean_dir}/{action_string}/variancepart/TEMP_hemi-{hemisphere}_mean.gii').agg_data()

    # find voxels greater than other features ______________________________
    hemi_dict[hemisphere]['actions'] = np.sum((actions_gii > objects_gii) & 
        (actions_gii > scenes_gii) &
        (actions_gii > moten_gii)
        ) / actions_gii.shape

    hemi_dict[hemisphere]['scenes'] = np.sum((scenes_gii > objects_gii) & 
        (scenes_gii > actions_gii) &
        (scenes_gii > moten_gii)
        ) / actions_gii.shape

    hemi_dict[hemisphere]['objects'] = np.sum((objects_gii > actions_gii) & 
        (objects_gii > scenes_gii) &
        (objects_gii > moten_gii)
        ) / actions_gii.shape
    hemi_dict[hemisphere]['moten'] = np.sum((moten_gii > objects_gii) & 
        (moten_gii > scenes_gii) &
        (moten_gii > actions_gii)
        ) / actions_gii.shape


# %%

for feature in feature_dict.keys():
    print(f"{feature}")
    print(np.mean([hemi_dict['lh'][feature], hemi_dict['rh'][feature]]))