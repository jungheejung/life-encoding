import numpy as np
import mvpa2.suite as mv
import os

directory = '/dartfs-hpc/scratch/cara/models/niml/complete/'
for file in ['ws', 'aa','ha_testsubj']:
        lh = mv.niml.read(os.path.join(directory, '{0}.lh.niml.dset'.format(file)))
        rh = mv.niml.read(os.path.join(directory, '{0}.rh.niml.dset'.format(file)))
        df = np.concatenate((lh, rh), axis=1)
        print(file, df.shape)
        print(np.min(df[0,:]), np.percentile(df[0,:], 5), np.percentile(df[0,:], 95), np.max(df[0,:]))
        print(np.min(df[4,:]), np.percentile(df[4,:], 5), np.percentile(df[4,:], 95), np.max(df[4,:]))
        print(np.min(df[5,:]), np.percentile(df[5,:], 5), np.percentile(df[5,:], 95), np.max(df[5,:]))
