import numpy as np

n_splits = 40
result = 'comb_r'
alignment = 'ws'
test_subject = 'sub-rid000005'
test_run = 1
hemi = 'lh'

stack_data = []
for split in np.arange(n_splits):
    split_data = np.load(f'{result}_align-{alignment}_{test_subject}_'
                         f'run-{test_run}_roi-{split}_hemi-{hemisphere}.npy')
    stack_data.append(split_data)

stack_data = np.concatenate(stack_data) #check this for multidim data

## SAVE AS NP ARRAY AND GIFTI
