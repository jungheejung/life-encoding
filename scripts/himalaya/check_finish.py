import json
import numpy as np
from itertools import product
from os.path import exists

# Set up subject, hemisphere, run parameters
subjects = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006',
            'sub-rid000009', 'sub-rid000012', 'sub-rid000014',
            'sub-rid000017', 'sub-rid000019', 'sub-rid000024',
            'sub-rid000027', 'sub-rid000031', 'sub-rid000032',
            'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
            'sub-rid000037', 'sub-rid000038', 'sub-rid000041']

hemispheres = ['lh', 'rh']
test_runs = [1, 2, 3, 4]
rois = np.arange(40).astype(str)
alignment = 'ws'# 'ws' #'aa' #'ha_common'
params = list(product(subjects, hemispheres, test_runs, rois))
print(f"total number of jobs: {len(params)}")

# TODO: if file not exist, print
# 
job_nums = []
save_dir = f'/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/results/himalaya/{alignment}'
for job_num, (subject, hemisphere, test_run, roi) in enumerate(params):
    fname = (f'{save_dir}/bg-r2_align-{alignment}_{subject}_'
             f'run-{test_run}_roi-{roi}_hemi-{hemisphere}.npy')
    if not exists(fname):
        print(f"job: {job_num}, {(subject, hemisphere, test_run, roi)}")
        print(fname)
        job_nums.append(job_num)

print(",".join([str(s + 2) for s in job_nums]))
