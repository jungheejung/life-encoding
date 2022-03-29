import numpy as np
import mvpa2.suite as mv
import nibabel as nib
import os

# https://github.com/snastase/narratives/blob/master/code/gifti_io.py
def write_gifti(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    gda = nib.gifti.GiftiDataArray(data)
    gii.add_gifti_data_array(gda)
    nib.gifti.giftiio.write(gii, output_fn)

# CHANGE PARAMETER main_dir
main_dir = '/Volumes/life_data/'
for hemi in ['lh', 'rh']:

# test_ds = mv.niml.read('/dartfs/rc/lab/D/DBIC/DBIC/life_data/niml/ws.{0}.niml.dset'.format(half))
    test_ds = mv.niml.read(os.path.join(main_dir, 'niml', 'ws.{0}.niml.dset'.format(hemi)))
    test_ds.samples
    type(test_ds.samples) # numpy.ndarray

    test_ds.samples.shape # (19, 40962)
    medial_half =  np.sum(test_ds.samples[1:, :] != 0, axis=0) == 0
    np.save(os.path.join(main_dir, 'niml', 'fsaverage6_medial_{0}.npy'.format(hemi)), medial_half)

    write_gifti(medial_half.astype(float), os.path.join(main_dir,'niml','fsaverage6_medial_{0}.gii'.format(hemi)),
    os.path.join(main_dir, 'life_dataset', 'sub-rid000041_task-life_acq-412vol_run-04.{0}.tproject.gii'.format(hemi)))