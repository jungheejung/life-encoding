import numpy as np
import os

main_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding'
stack_dir = os.path.join(main_dir, 'results', 'himalaya', 'ha_common')
n_splits = 40
result = "split-r" # ridge-coef comb-r split-r comb-r2 split-r2 comb-pred split-pred
alignment = 'ha_common'
#test_subject = 'sub-rid000005'
#test_run = 1
#hemispheres = 'lh'
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
                stack_data = {'bg': [], 'actions': [], 'agents': []}
                for split in np.arange(n_splits):
                    split_data = np.load(f'{stack_dir}/{result}_align-{alignment}_{test_subject}_'
                                f'run-{test_run}_roi-{split}_hemi-{hemisphere}.npy', allow_pickle=True).item()
                for feature in stack_data:
                    stack_data[feature].append(split_data[feature])

                for feature in stack_data:
                    split_result = f"{feature}-{result.split('-')[1]}"
                    split_f = (f'{stack_dir}/{split_result}_align-{alignment}_{test_subject}_'
                               f'run-{test_run}_hemi-{hemisphere}.npy')
                    stack_result = np.concatenate(stack_data[feature], axis=1)
                    print(f"stack_data shape: {stack_result.shape}")
                    np.save(split_f, stack_result)

            else:
                stack_data = []
                for split in np.arange(n_splits):
                    split_data = np.load(f'{stack_dir}/{result}_align-{alignment}_{test_subject}_'
                                f'run-{test_run}_roi-{split}_hemi-{hemisphere}.npy')
                    stack_data.append(split_data)
                stack_data = np.concatenate(stack_data, axis =1) #check this for multidim data
                print(f"stack_data shape: {stack_data.shape}")
                np.save(f"{stack_dir}/{result}_align-{alignment}_{test_subject}_run-{test_run}_hemi-{hemisphere}.npy", stack_data)

            ## SAVE AS NP ARRAY AND GIFTI
            print(f"finished recomining vertices: {test_subject} testrun-{test_run} {hemisphere}")
