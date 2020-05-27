import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
import os


models = ['ws', 'aa', 'ha_testsubj']
runs = range(1,5)
participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']

hemispheres = ['lh', 'rh']
data_dir = '/dartfs-hpc/scratch/cara/models/singlealpha/'


alpha_dict = {}
for m in models:
    for r in runs:
        for h in hemispheres:
            for p in participants:
                ds = np.load(os.path.join(data_dir, '{0}-leftout{1}_singlealpha/{2}/{3}/alphas.npy'.format(m, r, p, h)))
                alph = np.unique(ds)[0]
                if alph in alpha_dict:
                    alpha_dict[alph].add((p,m))
                else:
                    alpha_dict[alph] = set()
                    alpha_dict[alph].add((p,m))
