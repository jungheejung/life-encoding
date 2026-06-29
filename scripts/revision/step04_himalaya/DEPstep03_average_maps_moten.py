import numpy as np
import os, sys
from os.path import join
import nibabel as nib
import argparse
from pathlib import Path
"""
avg_all: average across all 4 runs (within run, average across participants)
avg_run: average within each run (across participants)
"""

def write_gifti(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    for row in data:
        gda = nib.gifti.GiftiDataArray(row)
        gii.add_gifti_data_array(gda)
    nib.gifti.giftiio.write(gii, output_fn)

def format_features(items):
    if len(items) == 2:
        return f"{items[0]}-{items[1]}"
    elif len(items) == 1:
        return items[0]
    else:
        return ""

parser = argparse.ArgumentParser()

parser.add_argument("--slurm-id", 
                    type=int, help="slurm id in numbers")
parser.add_argument("--align", choices=['aa', 'ws', 'ha_common'],
                    type=str, help="specify alignment of anatomical, within subject, or hyperalignment common")
# parser.add_argument("--analysis",  choices=['moten', 'base', 'pca', '0tr', 'glove_single'],
#                     type=str, help="features: 1) using base 300 features 2) PC extracted features 3) PC extracted features + motion energy 4) full moten but shifted TR")
parser.add_argument("--features",  choices=["actions moten","agents moten", "bg moten"],
                    type=str, help="combined features")
parser.add_argument("--pca", choices=[40, 60],
                    type=int, help="number of pcs")
args = parser.parse_args()

index = args.slurm_id # 'ws', 'aa', 'ha_test', 'ha_common'
alignment = args.align # 'lh' or 'rh'
# analysis = args.analysis
feature_args = args.features
pca_comp = args.pca

print(f"{alignment} {pca_comp} ")
suma_dir = '/Users/h/suma-fsaverage6'
main_dir = '/dartfs/rc/lab/H/HaxbyLab/heejung'

vast_dir = '/vast/labs/DBIC/datasets/Life/life-encoding' 
result_dir = '/dartfs/rc/lab/H/HaxbyLab/heejung/results/revision/glove/ws_pca-40'
#'/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding'

feature_args = args.features
features = format_features(feature_args)

output_dir = os.path.join(main_dir, 'results', 'revision', 'glove_nested', f'{alignment}_pca-{pca_comp}')

print(output_dir)
n_splits = 40

result_list = [ 'scenes-r', 'agents-r', 'actions-r','moten-r', 'objects-r', 'ridge-coef', 'comb-r','comb-r2', 'split-pred', 'split-r2', 'scenes-r2', 'agents-r2', 'actions-r2', 'moten-r2', 'objects-r2', 'comb-pred']
result = result_list[index]
import re


unique_keys = [
    re.search(r'sub-\d+', p.name).group()
    for p in Path(result_dir).glob('*.npy')
    if p.is_file() and re.search(f'sub-\d+', p.name)
]

# subjects = Path(result_dir).rglob('*sub-*.npy')
subjects = sorted(list(set(unique_keys)))
# subjects = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006',
#             'sub-rid000009', 'sub-rid000012', 'sub-rid000014',
#             'sub-rid000017', 'sub-rid000019', 'sub-rid000024',
#             'sub-rid000027', 'sub-rid000031', 'sub-rid000032',
#             'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
#             'sub-rid000037', 'sub-rid000038', 'sub-rid000041']
subjects = ['sub-0002', 'sub-0008', 'sub-0010', 'sub-0011', 'sub-0013', 'sub-0014', 'sub-0016', 'sub-0018', 'sub-0019', 'sub-0020', 'sub-0026', 'sub-0031', 'sub-0032', 'sub-0033', 'sub-0034', 'sub-0036', 'sub-0037', 'sub-0038', 'sub-0040', 'sub-0043', 'sub-0044', 'sub-0046', 'sub-0050', 'sub-0051', 'sub-0052', 'sub-0053', 'sub-0055', 'sub-0058', 'sub-0059', 'sub-0060', 'sub-0062', 'sub-0064', 'sub-0065', 'sub-0066', 'sub-0069', 'sub-0070', 'sub-0073', 'sub-0075', 'sub-0076', 'sub-0077', 'sub-0078', 'sub-0079', 'sub-0080', 'sub-0084', 'sub-0086', 'sub-0087', 'sub-0088', 'sub-0089', 'sub-0090', 'sub-0092', 'sub-0093', 'sub-0095', 'sub-0099', 'sub-0101', 'sub-0102', 'sub-0104', 'sub-0106', 'sub-0107', 'sub-0109', 'sub-0111', 'sub-0112', 'sub-0115', 'sub-0116', 'sub-0122', 'sub-0124', 'sub-0126', 'sub-0127', 'sub-0129', 'sub-0130', 'sub-0131', 'sub-0132', 'sub-0133']

runs = [1,2,3,4]
hemis = ['lh', 'rh']
for hemisphere in hemis:
    avg_all = []
    for test_run in runs:
        avg_run = []
        for test_subject in subjects:
            try:        
                split_fname = f"{output_dir}/{result}/{result}_pca-{pca_comp}_align-{alignment}_{test_subject}_run-{test_run}_hemi-{hemisphere}.npy"
                run_data = np.load(split_fname)
            except (FileNotFoundError) as e:
                print(f"missing file: {split_fname}")
                continue
            avg_run.append(run_data)
        if result[-1] == 'r': # split-r combine-r
            avg_run = np.tanh(np.mean(np.vstack(np.arctanh(avg_run)), axis = 0))
        else:
            avg_run = np.mean(np.vstack(avg_run), axis = 0)                
        np.save(f"{output_dir}/{result}_pca-{pca_comp}_align-{alignment}_sub-mean_run-{test_run}_hemi-{hemisphere}.npy", avg_run)
        write_gifti(np.nan_to_num(avg_run.astype(float)),
                    template_fn = f'/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml/fsaverage6_medial_{hemisphere}.gii',  
                    output_fn = join(output_dir,f'{result}_pca-{pca_comp}_align-{alignment}_sub-mean_run-{test_run}_hemi-{hemisphere}.gii'))
        avg_all.append(avg_run)
        print(f"finished averaging subjects for run {test_run}")
    if result[-1] == 'r':
        avg_all =  np.tanh(np.mean(np.vstack(np.arctanh(avg_all)), axis = 0))
    else:
        avg_all = np.mean(np.vstack(avg_all), axis = 0)
    np.save(f"{output_dir}/{result}_pca-{pca_comp}_align-{alignment}_sub-mean_run-mean_hemi-{hemisphere}.npy", avg_all)
    write_gifti(np.nan_to_num(avg_all.astype(float)),
                template_fn = f'/dartfs/rc/lab/H/HaxbyLab/heejung/data/niml/fsaverage6_medial_{hemisphere}.gii', 
                output_fn = join(output_dir,f'{result}_pca-{pca_comp}_align-{alignment}_sub-mean_run-mean_hemi-{hemisphere}.gii'))
    print(f"finished recomining vertices:  testrun-{test_run} {hemisphere}")
