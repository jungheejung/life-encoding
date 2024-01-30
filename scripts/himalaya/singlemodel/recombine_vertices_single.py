import numpy as np
import os, sys
import argparse

main_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding'

parser = argparse.ArgumentParser()

parser.add_argument("--slurm-id", 
                    type=int, help="slurm id in numbers")
parser.add_argument("--align", choices=['aa', 'ws', 'ha_common'],
                    type=str, help="specify alignment of anatomical, within subject, or hyperalignment common")
parser.add_argument("--analysis",  choices=['moten', 'base', 'pca', 'single'],
                    type=str, help="features: 1) using base 300 features 2) PC extracted features 3) PC extracted features + motion energy")
parser.add_argument("-f", "--features", nargs="*", type=str,
                    default=['bg', 'actions', 'agents'],
                    help="specify one or more feature spaces")
parser.add_argument("--pca", choices=[40, 60],
                    type=int, help="number of pcs")
args = parser.parse_args()

index = args.slurm_id # 'ws', 'aa', 'ha_test', 'ha_common'
alignment = args.align # 'lh' or 'rh'
features = args.features # e.g. ['bg', 'actions', 'agents', 'moten'] 
analysis_type = args.analysis
n_components = args.pca
pca = n_components

# alignment = sys.argv[2]
# analysis_type = 'moten' # 'moten', 'base', 'pca'

# stack_dir = os.path.join(main_dir, 'results', 'himalaya', analysis_type, alignment_pca)
stack_dir = os.path.join(main_dir, 'results', 'himalaya', f'single-{"".join(features)}', f'{alignment}_pca-{n_components}')
n_splits = 40
# index = int(sys.argv[1])
result_list = [ 'ridge-coef', 'split-r',  'comb-r', 'comb-r2', 'comb-pred', 'split-pred','split-r2']
result = result_list[index]
#result = 'ridge-coef' #"split-r" # ridge-coef comb-r split-r comb-r2 split-r2 comb-pred split-pred
print(result)
subjects = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006',
            'sub-rid000009', 'sub-rid000012', 'sub-rid000014',
            'sub-rid000017', 'sub-rid000019', 'sub-rid000024',
            'sub-rid000027', 'sub-rid000031', 'sub-rid000032',
            'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
            'sub-rid000037', 'sub-rid000038', 'sub-rid000041']
runs = [1,2,3,4]
hemis = ['lh', 'rh']

for test_subject in subjects:
    for test_run in runs:
        for hemisphere in hemis:
            if 'split' in result:
                stack_data = {k: [] for k in features}
                # stack_data = {'bg': [], 'moten': []} #  'actions': [], 'agents': [],
                for split in np.arange(n_splits):
                    split_data = np.load(f'{stack_dir}/{result}_feature-{"".join(features)}_pca-{pca}_align-{alignment}_{test_subject}_run-{test_run}_roi-{split}_hemi-{hemisphere}.npy', allow_pickle=True).item()
                    for feature in features:
                        stack_data[feature].append(split_data[feature])

                for feature in features:
                    split_result = f"{feature}-{result.split('-')[1]}"
                    split_f = (f'{stack_dir}/{split_result}_feature-{"".join(features)}_pca-{pca}_align-{alignment}_{test_subject}_run-{test_run}_hemi-{hemisphere}.npy')
                    stack_result = np.concatenate(stack_data[feature], axis=1)
                    print(f"stack_data shape: {stack_result.shape}")
                    np.save(split_f, stack_result)
            else:
                stack_data = []
                for split in np.arange(n_splits):
                    split_data = np.load(f'{stack_dir}/{result}_feature-{"".join(features)}_pca-{pca}_align-{alignment}_{test_subject}_run-{test_run}_roi-{split}_hemi-{hemisphere}.npy')
                    stack_data.append(split_data)
                stack_data = np.concatenate(stack_data, axis =1) #check this for multidim data
                print(f"stack_data shape: {stack_data.shape}")
                np.save(f"{stack_dir}/{result}_feature-{''.join(features)}_pca-{pca}_align-{alignment}_{test_subject}_run-{test_run}_hemi-{hemisphere}.npy", stack_data)

            ## SAVE AS NP ARRAY AND GIFTI
            print(f"finished recombining vertices {''.join(features)}: {test_subject} testrun-{test_run} {hemisphere}")
