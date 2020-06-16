#!/usr/bin/env python

# Compute paired t-test between anatomical and hyperaligned model fits

import numpy as np
import mvpa2.suite as mv
from scipy.stats import norm, ttest_1samp
from statsmodels.stats.multitest import multipletests

# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for hemi in ['lh', 'rh']:
    ws = mv.niml.read('ws_1.{0}.niml.dset'.format(hemi))
    cortical_vertices[hemi] = np.where(np.sum(ws.samples[1:, :] != 0, axis=0) != 0)[0].tolist()
    #cortical_vertices[hemi] = np.ones((40962))
    #cortical_vertices[hemi][np.sum(ws.samples[1:, :] != 0, axis=0) == 0] = 0

# The "tests" we want to do
tests = ['aa-ws', 'ha-ws', 'ha-aa']
hemis = ['lh', 'rh']

# Load data and extract individual subject maps
runwise_fits = {}
runwise_diffs = {}
results = {}
for hemi in hemis:
    runwise_fits[hemi] = {'ws': [], 'aa': [], 'ha': []}
    runwise_diffs[hemi] = {'diff(aa-ws)': [], 'diff(ha-ws)': [], 'diff(ha-aa)': []}
    results[hemi] = {}
    for run in [1, 2, 3, 4]:
        ws = mv.niml.read('ws_{0}.{1}.niml.dset'.format(run, hemi)).samples[1:, cortical_vertices[hemi]]
        aa = mv.niml.read('aa_{0}.{1}.niml.dset'.format(run, hemi)).samples[1:, cortical_vertices[hemi]]
        ha = mv.niml.read('ha_testsubj_{0}.{1}.niml.dset'.format(run, hemi)).samples[1:, cortical_vertices[hemi]]

        runwise_fits[hemi]['ws'].append(ws)
        runwise_fits[hemi]['aa'].append(aa)
        runwise_fits[hemi]['ha'].append(ha)

        # Compute runwise differences, then average them
        runwise_diffs[hemi]['diff(aa-ws)'].append(np.arctanh(aa) - np.arctanh(ws))
        runwise_diffs[hemi]['diff(ha-ws)'].append(np.arctanh(ha) - np.arctanh(ws))
        runwise_diffs[hemi]['diff(ha-aa)'].append(np.arctanh(ha) - np.arctanh(aa))
    for test in tests:
        runwise_diffs[hemi]['mean({0})'.format(test)] = np.mean(runwise_diffs[hemi]['diff({0})'.format(test)], axis=0)

    # Compute mean, t, p, z, q, and z(q) for our comparisons
    for test in tests:
        # Compute mean of paired differences
        results[hemi]['mean({0})'.format(test)] = np.mean(runwise_diffs[hemi]['mean({0})'.format(test)], axis=0)

        # Paired t-test (requires subjects in same order)
        results[hemi]['t({0})'.format(test)], results[hemi]['p({0})'.format(test)] = ttest_1samp(
            runwise_diffs[hemi]['mean({0})'.format(test)], 0, axis=0)

        # Get rid of nans from all-zero vertices
        #t = np.nan_to_num(t)
        #p = np.nan_to_num(p)

        # Convert p-values to z-values (set threshold to 1.96 for two-tailed test)
        results[hemi]['z({0})'.format(test)] = norm.ppf(1 - results[hemi]['p({0})'.format(test)])

        # Zero out 'inf' z-values resulting from p-values of 0
        #z[z == np.inf] = 0

# Compute FDR q-values to correct for multiple tests (w/o medial wall)
for test in tests:
    p_both = np.hstack((results['lh']['p({0})'.format(test)],
                        results['rh']['p({0})'.format(test)]))
    assert len(p_both) == len(cortical_vertices['lh']) + len(cortical_vertices['rh'])
    q_both = multipletests(p_both, alpha=.05, method='fdr_bh')[1]

    results['lh']['q({0})'.format(test)] = q_both[:len(cortical_vertices['lh'])]
    results['rh']['q({0})'.format(test)] = q_both[len(cortical_vertices['lh']):]

# Compute z-values from q-values
for hemi in hemis:
    for test in tests:
        results[hemi]['q_z({0})'.format(test)] = norm.ppf(1 - results[hemi]['q({0})'.format(test)])

# Stack all these up, reinsert medial wall, and save
for hemi in hemis:
    for test in tests:
        stack = []
        subtests = ['mean({0})'.format(test), 't({0})'.format(test),
                    'p({0})'.format(test), 'z({0})'.format(test),
                    'q({0})'.format(test), 'q_z({0})'.format(test)]
        for subtest in subtests:
            result = np.zeros(40962)
            np.put(result, cortical_vertices[hemi], results[hemi][subtest])
            stack.append(result)
        stack = np.vstack(stack)
        assert stack.shape == (6, 40962)
        mv.niml.write('group_diff_{0}.{1}.niml.dset'.format(test, hemi), stack)

# Compute proportion greater per subject
proportions = {}
for test in tests:
    proportions[test] = {}
    proportions[test]['all'] = (np.sum(np.hstack((
                                    runwise_diffs['lh']['mean({0})'.format(test)],
                                    runwise_diffs['rh']['mean({0})'.format(test)])) > 0, axis=1) /
                                float(len(cortical_vertices['lh']) + len(cortical_vertices['rh'])))
    proportions[test]['mean'] = np.mean(proportions[test]['all'])
    proportions[test]['t-value'], proportions[test]['p-value'] = ttest_1samp(proportions[test]['all'], .5)

# Mean difference
overall = {}
for fit in ['ws', 'aa', 'ha']:
    overall[fit] = {}
    overall[fit]['all'] = np.median(np.hstack((np.mean(runwise_fits['lh'][fit], axis=0),
                                      np.mean(runwise_fits['rh'][fit], axis=0))), axis=1)
    overall[fit]['mean'] = np.mean(overall[fit]['all'])
for test in tests:
    overall[test] = {}
    overall[test]['diff'] = np.mean(np.hstack((
                                    runwise_diffs['lh']['mean({0})'.format(test)],
                                    runwise_diffs['rh']['mean({0})'.format(test)])), axis=1)
    overall[test]['mean'] = np.mean(overall[test]['diff'])
    overall[test]['t-value'], overall[test]['p-value'] = ttest_1samp(overall[test]['diff'], 0)
