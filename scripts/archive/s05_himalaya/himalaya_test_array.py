#!/usr/bin/env python3


import os, sys, shutil
import argparse
import subprocess
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import PredefinedSplit
from himalaya.ridge import GroupRidgeCV
from himalaya.scoring import correlation_score
import nibabel as nib
import json

# Assign/create directories
user_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f000fb6'
scratch_dir = os.path.join('/dartfs-hpc/scratch', user_dir)
data_dir = '/dartfs/rc/lab/D/DBIC/DBIC/life_data/'

# Create scratch directory if it doesn't exist
if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)

# Hyperalignment directory (/idata/DBIC/cara/life/pymvpa/)
hyper_dir = os.path.join(data_dir, 'hyperalign_mapper')

# Life fMRI data directory (/idata/DBIC/snastase/life)
fmri_dir = os.path.join(data_dir, 'life_dataset')

# Semantic model directory (/idata/DBIC/cara/w2v/w2v_features)
model_dir = os.path.join(data_dir, 'w2v_feature')

# Set up some hard-coded experiment variables
subjects = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006',
            'sub-rid000009', 'sub-rid000012', 'sub-rid000014',
            'sub-rid000017', 'sub-rid000019', 'sub-rid000024',
            'sub-rid000027', 'sub-rid000031', 'sub-rid000032',
            'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
            'sub-rid000037', 'sub-rid000038', 'sub-rid000041']

tr = 2.5
model_durs = {1: 369, 2: 341, 3: 372, 4: 406}
fmri_durs = {1: 374, 2: 346, 3: 377, 4: 412}
n_samples = np.sum(list(fmri_durs.values()))
n_vertices = 40962
n_proc = 32 # how many cores do we have?
n_medial = {'lh': 3486, 'rh': 3491}
n_blocks = 3 #20# 8 hr instead of 30 min ... break up n_vertices
model_ndim = 300
run_labels = [1, 2, 3, 4]
roi_json = os.path.join(user_dir, 'rois.json')

# Parameters for optional PCA
run_pca = False
n_components = 30

# Parameters from job submission script
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--alignment", choices=['ws', 'aa', 'ha'],
                    help="within-subject, anatomical, or hyperalignment")
parser.add_argument("--hemisphere", choices=['lh', 'rh'],
                    help="specify hemisphere")
parser.add_argument("-r", "--test-run", choices=[1, 2, 3, 4],
                    type=int, help="specify test run")
parser.add_argument("-s", "--test-subject", help="specify test subject")
parser.add_argument("--roi", default=None,
                    help="ROI label from rois.json")
parser.add_argument("-f", "--features", nargs="*", type=str,
                    default=['bg', 'actions', 'agents'],
                    help="specify one or more feature spaces")
args = parser.parse_args()
print(args)

