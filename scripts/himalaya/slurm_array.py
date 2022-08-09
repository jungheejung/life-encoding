import json
import numpy as np
from itertools import product


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

params = list(product(subjects, hemispheres, test_runs, rois))

with open('slurm_array.txt', 'w') as f:
    f.writelines([f'{s},{h},{t},{r}\n' for s, h, t, r in params])

