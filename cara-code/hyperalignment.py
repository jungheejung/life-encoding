#! /usr/bin/env python
# OMP_NUM_THREADS=1 python hyperalignment.py > logs/searchlight.log 2>&1

import sys
from os import mkdir
from os.path import exists, join
from subprocess import call
import numpy as np
import mvpa2.suite as mv

base_dir = '/ihome/cara/life'
data_dir = '/idata/DBIC/snastase/life/'
suma_dir = '/idata/DBIC/snastase/life/SUMA'
mvpa_dir = '/idata/DBIC/cara/life/pymvpa'
if not exists(mvpa_dir):
    mkdir(mvpa_dir)
tr = {1:374, 2:346, 3:377, 4:412}


participants = sorted(['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041'], reverse=True)

hemispheres = ['lh', 'rh']

n_vertices = 40962
n_proc = 32
n_medial = {'lh': 3486, 'rh': 3491}

def load_data(filename):
    ds = mv.gifti_dataset(filename)
    ds.sa.pop('intents')
    ds.sa['subjects'] = [participant] * ds.shape[0]
    ds.fa['node_indices'] = range(n_vertices)

    # Z-score features across samples
    mv.zscore(ds, chunks_attr=None)
    return ds

for left_out in [4]:
    included = [1,2,3,4]
    included.remove(left_out)
    print("inc",  included)

    for hemi in ['rh']:
        # Load surface and create searchlight query engine
        surf = mv.surf.read(join(suma_dir, '{0}.pial.gii'.format(hemi)))
        qe = mv.SurfaceQueryEngine(surf, 20.0, distance_metric='dijkstra')
        print("Finished creating surface-based searchlight query engine")

        # Load in surface data sets
        dss = []
        for participant in participants:
            print(participant)
            print(tr[included[0]])
            ra = load_data(join(data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[included[0]], included[0], hemi)))
            rb = load_data(join(data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[included[1]], included[1], hemi)))
            rc = load_data(join(data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[included[2]], included[2], hemi)))

            ds = mv.vstack((ra, rb, rc))
            print(ds.samples.shape)
            dss.append(ds)

        n_samples = ds.samples.shape[0]
        # Exclude medial wall
        print(np.where(np.sum(ds.samples == 0, axis=0) == n_samples))
        medial_wall = np.where(np.sum(ds.samples == 0, axis=0) == n_samples)[0].tolist()
        print(len(medial_wall))
        cortical_vertices = np.where(np.sum(ds.samples == 0, axis=0) < n_samples)[0].tolist()
        assert len(medial_wall) == n_medial[hemi]
        assert len(medial_wall) + len(cortical_vertices) == n_vertices

        # Estimate searchlight hyperalignment transformation on movie data
        sl_hyper = mv.SearchlightHyperalignment(queryengine=qe, nproc=n_proc,
                                                nblocks=n_proc*8, featsel=1.0,
                                                mask_node_ids=cortical_vertices,
                                                tmp_prefix='/fastscratch/cara/tmpsl')

        print("Estimated transformation!")

        mv.debug.active += ['HPAL', 'SLC']
        mappers = sl_hyper(dss)
        print("Finished creating hyperalignment mappers!")

        # Organize and save fitted hyperalignment mappers
        assert len(participants) == len(mappers)
        mappers = {participant: mapper for participant, mapper
                   in zip(participants, mappers)}
        print("Reorganized hyperalignment mappers")

        mv.h5save(join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}_reverse.hdf5'.format(hemi, left_out)), mappers)
        print("Successfully saved hyperalignment mappers for left out run {0}".format(left_out))
