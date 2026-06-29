import numpy as np
import os, sys
import argparse
import re
from pathlib import Path
main_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding'

parser = argparse.ArgumentParser()

parser.add_argument("--slurm-id", 
                    type=int, help="slurm id in numbers")
parser.add_argument("--align", choices=['aa', 'ws', 'ha_common'],
                    type=str, help="specify alignment of anatomical, within subject, or hyperalignment common")
parser.add_argument("--analysis",  choices=['moten', 'base', 'pca', '0tr', 'glove'],
                    type=str, help="features: 1) using base 300 features 2) PC extracted features 3) PC extracted features + motion energy 4) full moten but shifted TR")
parser.add_argument("--pca", choices=[40, 60],
                    type=int, help="number of pcs")
args = parser.parse_args()

index = args.slurm_id # 'ws', 'aa', 'ha_test', 'ha_common'
alignment = args.align # 'ws'
analysis_type = args.analysis # glove
pca = args.pca

#################### interactive node ####################
# alignment = sys.argv[2]
# analysis_type = 'moten' # 'moten', 'base', 'pca'
alignment = 'ws'
pca = 40
alignment_pca = f'{alignment}_pca-{pca}'

stack_dir = '/dartfs/rc/lab/H/HaxbyLab/heejung/results/revision/glove/ws_pca-40'
# stack_dir = os.path.join(main_dir, 'results', 'himalaya', analysis_type, alignment_pca)
n_splits = 40
# index = int(sys.argv[1])
result_list = [ 'ridge-coef', 'split-r',  'comb-r', 'comb-r2', 'comb-pred', 'split-pred','split-r2']
result = result_list[index]
#result = 'ridge-coef' #"split-r" # ridge-coef comb-r split-r comb-r2 split-r2 comb-pred split-pred

unique_keys = [
    re.search(r'sub-\d+', p.name).group()
    for p in Path(stack_dir).glob('*.npy')
    if p.is_file() and re.search(f'sub-\d+', p.name)
]

print(result)
subjects = sorted(list(set(unique_keys)))
# subjects = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006',
#             'sub-rid000009', 'sub-rid000012', 'sub-rid000014',
#             'sub-rid000017', 'sub-rid000019', 'sub-rid000024',
#             'sub-rid000027', 'sub-rid000031', 'sub-rid000032',
#             'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
#             'sub-rid000037', 'sub-rid000038', 'sub-rid000041']
runs = [1,2,3,4]
hemis = ['lh', 'rh']

for test_subject in subjects:
    for test_run in runs:
        for hemisphere in hemis:
            if 'split' in result:
                stack_data = {'scenes': [], 'actions': [], 'agents': [], 'objects': [], 'moten': []}
                # for split in np.arange(n_splits):
                try: 
                    split_fname = f'{stack_dir}/{result}_pca-{pca}_align-{alignment}_{test_subject}_run-{test_run}_roi-None_hemi-{hemisphere}.npy'
                    split_data = np.load(split_fname, allow_pickle=True).item()
                except (FileNotFoundError) as e:
                    print(f"missing file: {split_fname}")
                    continue
                for feature in stack_data:
                    stack_data[feature] = split_data[feature]

                for feature in stack_data:
                    split_result = f"{feature}-{result.split('-')[1]}"
                    Path(f'{stack_dir}/{split_result}').mkdir(parents=True, exist_ok=True)
                    split_f = (f'{stack_dir}/{split_result}/{split_result}_pca-{pca}_align-{alignment}_{test_subject}_run-{test_run}_hemi-{hemisphere}.npy')
                    # stack_result = np.concatenate(stack_data[feature], axis=1)
                    # print(f"stack_data shape: {stack_data.shape}")
                    np.save(split_f, stack_data[feature]) #stack_result)
            else:
                stack_data = []
                # for split in np.arange(n_splits):
                nonsplit_fname = f'{stack_dir}/{result}_pca-{pca}_align-{alignment}_{test_subject}_run-{test_run}_roi-None_hemi-{hemisphere}.npy'
                try:
                    split_data = np.load(nonsplit_fname)
                except (FileNotFoundError) as e:
                    print(f"missing file: {nonsplit_fname}")
                    continue

                # stack_data.append(split_data)
                stack_data = np.concatenate(stack_data, axis =1) #check this for multidim data
                print(f"stack_data shape: {stack_data.shape}")
                Path(f'{stack_dir}/{result}/{result}').mkdir(parents=True, exist_ok=True)
                np.save(f"{stack_dir}/{result}/{result}_pca-{pca}_align-{alignment}_{test_subject}_run-{test_run}_hemi-{hemisphere}.npy", stack_data)

            ## SAVE AS NP ARRAY AND GIFTI
            print(f"finished recombining vertices: {test_subject} testrun-{test_run} {hemisphere}")
