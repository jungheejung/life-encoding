#!/usr/bin/env python2

from os.path import join
import mvpa2 as mv
import nibabel as nib

niml_dir = '/dartfs/rc/lab/D/DBIC/DBIC/life_data/niml'

hemi = 'lh'

test = mv.niml.read(join(niml_dir, 'ws.{0}.niml.dset'.format(hemi)))

print(test.shape)
