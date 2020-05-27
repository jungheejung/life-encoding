import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
from os.path import join

n_vertices = 40962
data_dir = '/dartfs-hpc/scratch/cara/models/singlealpha/sa_niml/isc'
# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_dat = mv.niml.read('/idata/DBIC/cara/life/ridge/models/new_niml/ws/ws_run1.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(test_dat.samples[1:, :] != 0, axis=0) == 0] = 0

for model in ['AA', 'HA']:
    for hemi in ['lh', 'rh']:
        npy = np.load('{0}_{1}_masked.npy'.format(model, hemi))

        med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
        ds = np.zeros((npy.shape[0]+med_wall_ind.shape[0]),dtype=npy.dtype)
        ds[cortical_vertices[hemi] == 1] = npy
        ds = ds[None,:]

        mv.niml.write(join(data_dir, '{0}_rg_isc.{1}.niml.dset'.format(model.lower(), hemi)), ds)
