import matplotlib; matplotlib.use('agg')

import mvpa2.suite as mv
import pandas as pd
import numpy as np
from scipy import stats
from utils import hyper_ridge
import sys, os, time, csv

data_dir = '/dartfs-hpc/scratch/cara/new_models'
models = ['all', 'actions', 'agents', 'bg', 'actions_agents', 'actions_bg', 'agents_bg']

participants = sorted(['sub-rid000037', 'sub-rid000001', 'sub-rid000033', 'sub-rid000024',
                'sub-rid000041', 'sub-rid000032', 'sub-rid000006',
                'sub-rid000009', 'sub-rid000017', 'sub-rid000005', 'sub-rid000038',
                'sub-rid000031', 'sub-rid000012', 'sub-rid000027', 'sub-rid000014',
                'sub-rid000034', 'sub-rid000036', 'sub-rid000019'])
hemispheres = ['lh', 'rh']
run = 4
n_vertices = 40962

for modality in ['visual_0vec']:
    data_dir = '/dartfs-hpc/scratch/cara/new_models/' + modality

    corrs = {}
    for model in models:
        lh = np.load(os.path.join(data_dir, 'npy', 'full_{0}_{1}_{2}.{3}.npy'.format(modality, model, run, 'lh')))
        rh = np.load(os.path.join(data_dir, 'npy', 'full_{0}_{1}_{2}.{3}.npy'.format(modality, model, run, 'rh')))
        corrs['{0}'.format(model)] = np.square(np.concatenate((lh, rh)))
        # print(corrs['{0}'.format(model)])

    corrs['agents_i_actions'] = np.subtract(np.add(corrs['agents'], corrs['actions']), corrs['actions_agents'])
    corrs['agents_i_bg'] = np.subtract(np.add(corrs['agents'], corrs['bg']), corrs['agents_bg'])
    corrs['bg_i_actions'] = np.subtract(np.add(corrs['bg'], corrs['actions']), corrs['actions_bg'])

    corrs['all_i'] = np.subtract(np.add(np.add(np.add(corrs['all'], corrs['actions']), corrs['agents']), corrs['bg']), np.add(np.add(corrs['actions_agents'], corrs['agents_bg']), corrs['actions_bg']))
    # corrs['agents_i_actions_nobg'] = np.subtract(np.add(corrs['agents'], corrs['actions']), np.add(corrs['actions_agents'], corrs['all_i']))
    # corrs['agents_i_bg_noactions'] = np.subtract(np.add(corrs['agents'], corrs['bg']), np.add(corrs['agents_bg'], corrs['all_i']))
    # corrs['bg_i_actions_noagents'] = np.subtract(np.add(corrs['bg'], corrs['actions']), np.add(corrs['actions_bg'], corrs['all_i']))

    RC_corrs = {}
    RC_corrs['only_actions_RC'] = np.add(np.subtract(corrs['actions'], np.add(corrs['bg_i_actions'], corrs['agents_i_actions'])), corrs['all_i'])

    RC_corrs['only_agents_RC'] = np.add(np.subtract(corrs['agents'], np.add(corrs['agents_i_bg'], corrs['agents_i_actions'])), corrs['all_i'])
    RC_corrs['only_bg_RC'] = np.add(np.subtract(corrs['bg'], np.add(corrs['bg_i_actions'], corrs['agents_i_bg'])), corrs['all_i'])
    data_dir = '/dartfs-hpc/scratch/cara/new_models/visual_0vec'

    np.save(os.path.join(data_dir, 'niml', 'variance_partition', 'only_actions.npy'), RC_corrs['only_actions_RC'])
    np.save(os.path.join(data_dir, 'niml', 'variance_partition', 'only_agents.npy'), RC_corrs['only_agents_RC'])
    np.save(os.path.join(data_dir, 'niml', 'variance_partition', 'only_bg.npy'), RC_corrs['only_bg_RC'])


    # First let's create mask of cortical vertices excluding medial wall
    cortical_vertices = {}
    for hemi in ['lh', 'rh']:
        test_dat = mv.niml.read('/idata/DBIC/cara/life/ridge/models/new_niml/ws/ws_run1.{0}.niml.dset'.format(hemi))
        cortical_vertices[hemi] = np.ones((n_vertices))
        cortical_vertices[hemi][np.sum(test_dat.samples[1:, :] != 0, axis=0) == 0] = 0

    for hemi in hemispheres:
        for RC in RC_corrs.keys():
            if hemi == 'lh':
                ds = RC_corrs[RC][:19,:]
            else:
                ds = RC_corrs[RC][19:,:]

            med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
            ds[:,med_wall_ind] = 0
            print(ds.shape)

            print(os.path.join(data_dir, 'niml', 'variance_partition', 'old_{0}.{1}.niml.dset'.format(RC, hemi)))
            mv.niml.write(os.path.join(data_dir, 'niml', 'variance_partition', 'old_{0}.{1}.niml.dset'.format(RC, hemi)), ds)
