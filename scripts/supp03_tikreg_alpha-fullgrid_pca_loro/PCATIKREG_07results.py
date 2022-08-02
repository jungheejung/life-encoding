
"""
We plot average correlation coefficients from the cv 
plot correlation coefficient x hyperparameter x feature x ROI

- [ ] plot hyperparameter and coefficient per participant
- [ ] plot sub plots per participant per run per roi
- [ ] for starters, just grab the first participant's hyperparameter x corr coefficient
"""


# 1. overview

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import glob
import itertools
import os

# %%
# directory
result_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/PCA_tikreg-loro_fullrange-10000-ROI/'
align = 'ws'
model = 'visual'

# %%
feature = 'actions'
hemi = 'lh'
roi = 'ips'
# total_avg = appended_data.groupby('run').mean().mean()
total_avg = pd.read_csv(os.path.join(result_dir, align, model, 'bg_actions_agents',
'corr-totalavg_feature-{0}_hemi-{1}_roi-{2}.csv'.format(feature, hemi, roi)))

# %%
column_indices = [0, 1]
new_names = ['node','correlation']
old_names = total_avg.columns[column_indices]
total_avg.rename(columns=dict(zip(old_names, new_names)), inplace=True)
total_avg.head()

# %% 
hyperp_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/PCA_tikreg-loro_fullrange-10000-ROI/ws/visual/bg_actions_agents/leftout_run_1/sub-rid000001/rh'
hyper_basename = 'hyperparam-alpha_sub-rid000001_model-visual_align-ws_foldshifted-1_hemi-rh_range-ips.json'
corr_basename = 'corrcoef_sub-rid000001_model-visual_align-ws_feature-actions_foldshifted-1_hemi-rh_range-ips.json'
hyper_fname = os.path.join(hyperp_dir, hyper_basename)
corr_basename= os.path.join(hyperp_dir, corr_basename)
with open(hyper_fname) as h:
    hyper = json.load(h)

with open(corr_basename) as c:
    corr = json.load(c)

hyper_df = pd.DataFrame(hyper)
corr_df = pd.DataFrame(corr)

# hyper_df.head()
column_indices = [0, 1]
new_names = ['node','hyperparameter']
old_names = hyper_df.columns[column_indices]
hyper_df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
hyper_df.head()

# %%
column_indices = [0, 1]
new_names = ['node','correlation']
old_names = corr_df.columns[column_indices]
corr_df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
corr_df.head()

# %%
df_merged = corr_df.merge(hyper_df, on=["node"])
df_merged.head()

# %% 
b = df_merged.plot.bar(x='node', y='hyperparameter', rot=45, )
ax = df_merged.plot.bar(x='node', y='correlation', rot=45,)

# %%

# ax2 = stack_df.plot.scatter(x='hyperparameter',
#                       y='corr',
#                       c='feature',
#                       colormap='viridis')

g =sns.scatterplot(x="hyperparameter", y="correlation",
              #hue="feature",
              data=df_merged);
g.set_xscale('log')

######################

# %%
# for loop load participant per run
hyperp_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/PCA_tikreg-loro_fullrange-10000-ROI/ws/visual/bg_actions_agents/leftout_run_1/sub-rid000001/rh'
hyper_basename = 'hyperparam-alpha_sub-rid000001_model-visual_align-ws_foldshifted-1_hemi-rh_range-ips.json'
corr_basename = 'corrcoef_sub-rid000001_model-visual_align-ws_feature-actions_foldshifted-1_hemi-rh_range-ips.json'

folders = glob.glob('/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/PCA_tikreg-loro_fullrange-10000-ROI/ws/visual/bg_actions_agents/*/sub-rid000001/rh')
folders


# print list(itertools.permutations([['lh', 'rh'], ['ips', 'loc', 'vt'], [1,2,3,4]]))
a = [['sub-rid000001', 'sub-rid000024'],[ 'rh'], ['ips', 'loc', 'vt'], [1,2,3,4]]
b = list(itertools.product(*a))

# %%
fig, axes = plt.subplots(len(b)/4, 4, figsize=(15, 10),  sharey=True, sharex = True)
fig.suptitle('sub-rid000001 - hyperparam & correlation per roi/run')

ind = 0
for sub, hemi, roi, run in b:

    # hemi = x[0];    roi = x[1];    run = x[2]
    hyperp_dir = os.path.join('/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/PCA_tikreg-loro_fullrange-10000-ROI/ws/visual/bg_actions_agents/', 'leftout_run_{0}'.format(run), sub, hemi)
    hyper_bname = 'hyperparam-alpha_{0}-visual_align-ws_foldshifted-{1}_hemi-{2}_range-{3}.json'.format(sub, run, hemi, roi)
    corr_bname = 'corrcoef_{0}-visual_align-ws_feature-actions_foldshifted-{1}_hemi-{2}_range-{3}.json'.format(sub, run, hemi, roi)
    hyper_fname = os.path.join(hyperp_dir, hyper_bname)
    corr_fname = os.path.join(hyperp_dir, corr_bname)
    info = os.path.basename(hyper_fname).split("_")
    with open(hyper_fname) as h:
        hyper = json.load(h)

    with open(corr_fname) as c:
        corr = json.load(c)

    hyper_df = pd.DataFrame(hyper)
    corr_df = pd.DataFrame(corr)

    column_indices = [0, 1]
    new_names = ['node','hyperparameter']
    old_names = hyper_df.columns[column_indices]
    hyper_df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    hyper_df.head()


    column_indices = [0, 1]
    new_names = ['node','correlation']
    old_names = corr_df.columns[column_indices]
    corr_df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    corr_df.head()

    df_merged = corr_df.merge(hyper_df, on=["node"])
    df_merged.head()



    g =sns.scatterplot(ax=axes[ind//4, run-1], x="hyperparameter", y= "correlation",
                #hue="feature",
                data=df_merged);
    g.set_title('run - {0}'.format(run))
    g.set_xscale('log')
    g.set_ylabel(str('{0} - corr'.format(roi.upper())))
    ind += 1
save_dir = os.path.join('/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/PCA_tikreg-loro_fullrange-10000-ROI/ws/visual/bg_actions_agents/')

g.savefig(os.path.join(save_dir, "hyperNcorr_{0}_hemi-{1}.png".format(sub, hemi)))
# %% 



