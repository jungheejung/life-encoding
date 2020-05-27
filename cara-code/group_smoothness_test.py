#!/usr/bin/env python

# Compute FWHM smoothness and perform t-test
import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
from subprocess import call, check_output
from scipy.stats import ttest_1samp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# First let's create mask of cortical vertices excluding medial wall
for hemi in ['lh', 'rh']:
    ws = mv.niml.read('ws_1.{0}.niml.dset'.format(hemi))
    cortical_vertices = np.ones((40962))
    cortical_vertices[np.sum(ws.samples[1:, :] != 0, axis=0) == 0] = 0
    mv.niml.write('cortical_vertices.{0}.niml.dset'.format(hemi), cortical_vertices[None, :])
    np.savetxt('cortical_vertices.{0}.1D'.format(hemi), cortical_vertices, fmt='%i', delimiter='\n')

# Estimate smoothness using AFNI/SUMA's SurfFWHM
smoothness = {}
for hemi in ['lh', 'rh']:
    smoothness[hemi] = {}
    for run in [1, 2, 3, 4]:
        smoothness[hemi][run] = {}
        for alignment in ['ws', 'aa', 'ha_testsubj']:
            smooths = check_output("SurfFWHM -clean "
                                   "-input {0}_{1}.{2}.niml.dset "
                                   "-i ~/life/forward_encoding/SUMA/{2}.pial.gii "
                                   "-b_mask cortical_vertices.{2}.1D".format(
                                       alignment, run, hemi),
                                   shell=True).split('\n')[1:19]
            smooths = [float(s) for s in smooths]
            smoothness[hemi][run][alignment] = np.array(smooths)

# Convert to long-form pandas format for plotting
df = {'Hemisphere': [], 'Run': [], 'Method': [], 'Subject': [], 'Smoothness (mm)': []}
for hemi in [('lh', 'L'), ('rh', 'R')]:
    for run in [1, 2, 3, 4]:
        for alignment in [('ws', 'Within-subject'), ('aa', 'Between-subject\n(anatomical)'), ('ha_testsubj', 'Between-subject\n(hyperalignment)')]:
            for subject in range(18):
                df['Hemisphere'].append(hemi[1])
                df['Run'].append(str(run))
                df['Method'].append(alignment[1])
                df['Subject'].append(subject + 1)
                df['Smoothness (mm)'].append(smoothness[hemi[0]][run][alignment[0]][subject])
df = pd.DataFrame(df)

sns.factorplot(x='Method', y='Smoothness (mm)', hue='Run',
               data=df, kind='bar', order=['Within-subject',
                                           'Between-subject\n(anatomical)',
                                           'Between-subject\n(hyperalignment)'],
               size=5, aspect=1.2, palette='muted')
plt.show
plt.savefig('smoothness_ana.png', dpi=300)

# Get means
means = {}
means['ws'] = np.mean(np.vstack(smoothness[hemi][run]['ws']
                                for hemi in ['lh', 'rh']
                                for run in [1, 2, 3, 4]))
means['aa'] = np.mean(np.vstack(smoothness[hemi][run]['aa']
                                for hemi in ['lh', 'rh']
                                for run in [1, 2, 3, 4]))
means['ha'] = np.mean(np.vstack(smoothness[hemi][run]['ha_testsubj']
                                for hemi in ['lh', 'rh']
                                for run in [1, 2, 3, 4]))

# t-test between alignments collapsing across runs
differences = {}
for hemi in ['lh', 'rh']:
    differences[hemi] = {}
    for run in [1, 2, 3, 4]:
        differences[hemi][run] = {}
        differences[hemi][run]['aa - ws'] = (smoothness[hemi][run]['aa'] -
                                             smoothness[hemi][run]['ws'])
        differences[hemi][run]['ha - ws'] = (smoothness[hemi][run]['ha_testsubj'] -
                                             smoothness[hemi][run]['ws'])
        differences[hemi][run]['aa - ha'] = (smoothness[hemi][run]['aa'] -
                                             smoothness[hemi][run]['ha_testsubj'])
mean_differences = {}
mean_differences['aa - ws'] = np.mean(np.vstack(differences[hemi][run]['aa - ws']
                                                for hemi in ['lh', 'rh']
                                                for run in [1, 2, 3, 4]), axis=0)
mean_differences['ha - ws'] = np.mean(np.vstack(differences[hemi][run]['ha - ws']
                                                for hemi in ['lh', 'rh']
                                                for run in [1, 2, 3, 4]), axis=0)
mean_differences['aa - ha'] = np.mean(np.vstack(differences[hemi][run]['aa - ha']
                                                for hemi in ['lh', 'rh']
                                                for run in [1, 2, 3, 4]), axis=0)
t_tests = {}
for test in mean_differences:
    t_tests[test] = ttest_1samp(mean_differences[test], 0)
