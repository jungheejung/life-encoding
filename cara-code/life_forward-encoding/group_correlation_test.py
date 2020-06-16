#!/usr/bin/env python

# Compute correlations between model fit maps
import matplotlib; matplotlib.use('agg')

import numpy as np
import mvpa2.suite as mv
from scipy.stats import pearsonr, ttest_rel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for hemi in ['lh', 'rh']:
    ws = mv.niml.read('ws_1.{0}.niml.dset'.format(hemi))
    cortical_vertices[hemi] = np.ones((40962))
    cortical_vertices[hemi][np.sum(ws.samples[1:, :] != 0, axis=0) == 0] = 0

# Get correlations between model fit maps per run
correlations = {}
for subject in range(18):
    correlations[subject] = {}
    for run in [1, 2, 3, 4]:
        correlations[subject][run] = {}
        ws = np.hstack((
            mv.niml.read('ws_{0}.lh.niml.dset'.format(run)).samples[1 + subject, cortical_vertices['lh'] == 1],
            mv.niml.read('ws_{0}.rh.niml.dset'.format(run)).samples[1 + subject, cortical_vertices['rh'] == 1]))
        aa = np.hstack((
            mv.niml.read('aa_{0}.lh.niml.dset'.format(run)).samples[1 + subject, cortical_vertices['lh'] == 1],
            mv.niml.read('aa_{0}.rh.niml.dset'.format(run)).samples[1 + subject, cortical_vertices['rh'] == 1]))
        ha = np.hstack((
            mv.niml.read('ha_testsubj_{0}.lh.niml.dset'.format(run)).samples[1 + subject, cortical_vertices['lh'] == 1],
            mv.niml.read('ha_testsubj_{0}.rh.niml.dset'.format(run)).samples[1 + subject, cortical_vertices['rh'] == 1]))
        correlations[subject][run]['corr(ws, aa)'] = pearsonr(ws, aa)
        correlations[subject][run]['corr(ws, ha)'] = pearsonr(ws, ha)

# Put these into a data frame and plot
df = {'Run': [], 'Alignment': [], 'Subject': [], 'Spatial correlation': []}
for subject in range(18):
    for run in [1, 2, 3, 4]:
        for comp in [('corr(ws, aa)', 'Anatomical alignment'), ('corr(ws, ha)', 'Hyperalignment')]:
            df['Run'].append(run)
            df['Subject'].append(subject + 1)
            df['Alignment'].append(comp[1])
            df['Spatial correlation'].append(correlations[subject][run][comp[0]][0])
df = pd.DataFrame(df)

sns.factorplot(x='Alignment', y='Spatial correlation', hue='Run',
               data=df, kind='bar', order=['Anatomical alignment',
                                           'Hyperalignment'],
               size=5, aspect=1, palette='muted')
plt.savefig('spatial_corr_vs_ws.png', dpi=300)

# # Put these into a data frame and plot ###INCLUDING COMMON SPACE
# df = {'Run': [], 'Alignment': [], 'Subject': [], 'Spatial correlation': []}
# for subject in range(18):
#     for run in [1, 2, 3, 4]:
#         for comp in [('corr(ws, aa)', 'Anatomical alignment'), ('corr(ws, ha)', 'Hyperalignment')]:
#             df['Run'].append(run)
#             df['Subject'].append(subject + 1)
#             df['Alignment'].append(comp[1])
#             df['Spatial correlation'].append(correlations[subject][run][comp[0]][0])
# df = pd.DataFrame(df)
#
# sns.factorplot(x='Alignment', y='Spatial correlation', hue='Run',
#                data=df, kind='bar', order=['Anatomical alignment',
#                                            'Hyperalignment'],
#                size=5, aspect=1.2, palette='muted')
# plt.savefig('spatial_corr_vs_WS.png', dpi=300)

# Compute t-test with Fisher Z transformation
t_value = ttest_rel(np.mean(np.vstack(
                            [[np.arctanh(correlations[s][r]['corr(ws, ha)'][0])
                              for s in range(18)]
                             for r in range(1, 5)]), axis=0),
                    np.mean(np.vstack(
                            [[np.arctanh(correlations[s][r]['corr(ws, aa)'][0])
                              for s in range(18)]
                             for r in range(1, 5)]), axis=0))
mean_diff = np.mean(np.mean(np.vstack(
                            [[correlations[s][r]['corr(ws, ha)'][0]
                              for s in range(18)]
                             for r in range(1, 5)]), axis=0) -
                    np.mean(np.vstack(
                            [[correlations[s][r]['corr(ws, aa)'][0]
                              for s in range(18)]
                             for r in range(1, 5)]), axis=0))
