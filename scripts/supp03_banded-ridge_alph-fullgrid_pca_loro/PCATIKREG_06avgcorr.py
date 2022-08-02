import re
import os
import pandas as pd
import numpy as np
import json
import glob
from collections import Iterable
import mvpa2.suite as mv
import sys

def flatten(coll):
    for i in coll:
            if isinstance(i, Iterable) and not isinstance(i, basestring):
                for subc in flatten(i):
                    yield subc
            else:
                yield i


# step 1 __________________________________________________________
result_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/PCA_tikreg-loro_fullrange-10000-ROI/'
align = 'ws'
model = "visual"
print(sys.argv)
feature = sys.argv[1]
hemi = sys.argv[2]
roi = sys.argv[3] #int(sys.argv[9])
# feature = "actions"
# hemi = "lh"
# roi = "ips"
ips_agent_flist = glob.glob(os.path.join(result_dir,align, model, 'bg_actions_agents' '/*/*/*/corr*{0}*_hemi-{1}*-{2}.json'.format(feature, hemi, roi)))
# list number of subject and runs
print("step 1: grab arguments and finished glob")
# ips_action_fname
# ips_agent_fname
# ips_bg_fname
# ips_hyper_fname



col_list = ['sub', 'run']
col_list.append(list(np.arange(40962)))
col_names = list(flatten(col_list))

full_df = pd.DataFrame(columns = col_names)
# step 2 __________________________________________________________
# 1) load json
appended_data = []
for ind, ips_agent_fpath in enumerate(sorted(ips_agent_flist)):
    ips_agent_fname  = os.path.basename(ips_agent_fpath)
    info = ips_agent_fname.split("_")
    p = info[1]
    feature = info[4].split("-")[-1]
    run = info[5].split("-")[-1]

    #with open(ips_action_json) as f:
      #coef_ips_action = json.load(f)
    with open(ips_agent_fpath) as f:
      coef_ips_agent = json.load(f)
    #with open(ips_bg_json) as f:
    #  coef_ips_bg = json.load(f)
    #with open(ips_hyper_json) as f:
    #  hyper_ips = json.load(f)

    # 2) convert to pandas
    #hyper_df = pd.DataFrame(hyper_ips)
    #actions_df = pd.DataFrame(coef_ips_action)
    agents_df = pd.DataFrame(coef_ips_agent)
    #bg_df = pd.DataFrame(coef_ips_bg)

    #hyper_df.rename(columns={0:'node', 1:'hyperparameter'}, inplace = True)
    agents_df.rename(columns={0:'node', 1:'corr'}, inplace = True)
    #bg_df.rename(columns={0:'node', 1:'corr'}, inplace = True)
    #actions_df.rename(columns={0:'node', 1:'corr'},  inplace = True)

    blocks = np.arange(40962)
    index_agents_df = pd.DataFrame(); index_agents_df['node'] = blocks.tolist(); index_agents_df['corr'] = 0
    #index_actions_df = pd.DataFrame(); index_actions_df['node'] = blocks.tolist(); index_actions_df['corr'] = 0
    #index_bg_df = pd.DataFrame(); index_bg_df['node'] = blocks.tolist(); index_bg_df['corr'] = 0

    for df_ind in range(len(agents_df)):
        key = int(agents_df.loc[df_ind, 'node'])
        index_agents_df.loc[key, 'corr'] =  agents_df.loc[df_ind, 'corr']

    # for df_ind in range(len(actions_df)):
    #     key = int(actions_df.loc[df_ind, 'node'])
    #     index_actions_df.loc[key, 'corr'] =  actions_df.loc[df_ind, 'corr']
    #
    # for df_ind in range(len(bg_df)):
    #     key = int(bg_df.loc[df_ind, 'node'])
    #     index_bg_df.loc[key, 'corr'] =  bg_df.loc[df_ind, 'corr']

    # individual p df
    X = index_agents_df.T
    X['sub'] = p
    X['run'] = run
    subset = X.drop('node').copy()

    # append to full dataframe
    appended_data.append(subset)
print("step 2: finished apppending data")

appended_data = pd.concat(appended_data)
appended_data.to_csv(os.path.join(result_dir, align, model, 'bg_actions_agents',
'corr-full_feature-{0}_hemi-{1}_roi-{2}.csv'.format(feature, hemi, roi)))
# save appended_data
run_avg = appended_data.groupby('run').mean()
run_avg.to_csv(os.path.join(result_dir, align, model, 'bg_actions_agents',
'corr-runavg_feature-{0}_hemi-{1}_roi-{2}.csv'.format(feature, hemi, roi)))
total_avg = appended_data.groupby('run').mean().mean()
total_avg.to_csv(os.path.join(result_dir, align, model, 'bg_actions_agents',
'corr-totalavg_feature-{0}_hemi-{1}_roi-{2}.csv'.format(feature, hemi, roi)), header = True)

print("saved as csv")


X = total_avg.values
print(X.shape)
final = np.reshape(X, (1,40962))
print(final.shape)
print("saving niml")
mv.niml.write(os.path.join(result_dir, align, model, 'bg_actions_agents',
'groupaverage_{0}_model-{1}_feature-{2}_roi-{3}.{4}.niml.dset'.format(align,model,feature,roi,hemi)), final)

    # subrunwise = np.array(index_agents_df['corr'])
    # np.save(os.path.join(group_dir, ips_agent_fname + '.npy', subrunwise)
    # mv.niml.write(os.path.join(group_dir, ips_agent_fname + '.niml.dset', subrunwise)
    # sort columns

# step 2 __________________________________________________________
# within run, subjectwise average
# group coor run 1
# group coor run 2
# save as niml


# step 3 __________________________________________________________
# save as niml

