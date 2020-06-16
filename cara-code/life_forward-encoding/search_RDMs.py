#!/usr/bin/env python

from sys import argv
from os.path import join
import numpy as np
import mvpa2.suite as mv
from scipy.spatial.distance import cdist, squareform
from mvpa2.measures import rsa

zscore_features = True
hyperalign = True

n_medial = {'lh': 3486, 'rh': 3491}
tr = {1:374, 2:346, 3:377, 4:412}

n_vertices = 40962

# base_dir = '/home/nastase/social_actions'
# scripts_dir = join(base_dir, 'scripts')
# data_dir = join(base_dir, 'fmri', '1021_actions', 'derivatives')
# suma_dir = join(data_dir, 'freesurfer', 'fsaverage6', 'SUMA')

mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
sam_data_dir = '/idata/DBIC/snastase/life'
suma_dir = join(sam_data_dir, 'SUMA')
ridge_dir = '/idata/DBIC/cara/life/ridge'
cara_data_dir = '/idata/DBIC/cara/life/data/'


participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']

hemispheres = ['lh', 'rh']

for hemi in hemispheres:
    # Load surface and create searchlight query engine
    surf = mv.surf.read(join(suma_dir, '{0}.pial.gii'.format(hemi)))
    qe = mv.SurfaceQueryEngine(surf, 20.0, distance_metric='dijkstra')
    print("Finished creating surface-based searchlight query engine")

    # Optional load hyperalignment mappers
    if hyperalign:
        mappers1 = mv.h5load(join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_1.hdf5'.format(hemi)))
        mappers2 = mv.h5load(join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_2.hdf5'.format(hemi)))
        mappers3 = mv.h5load(join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_3.hdf5'.format(hemi)))
        mappers4 = mv.h5load(join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_4.hdf5'.format(hemi)))
        mappers = [mappers1, mappers2, mappers3, mappers4]
        print("Loaded hyperalignment mappers")

for participant in participants:

    # Load in functional data
    ds1 = mv.gifti_dataset(join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[1], 1, hemi)))
    ds2 = mv.gifti_dataset(join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[2], 2, hemi)))
    ds3 = mv.gifti_dataset(join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[3], 3, hemi)))
    ds4 = mv.gifti_dataset(join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[4], 4, hemi)))

    # Exclude medial wall
    medial_wall = np.where(np.sum(ds1.samples == 0, axis=0) == tr[1])[0].tolist()
    cortical_vertices = np.where(np.sum(ds1.samples == 0, axis=0) < tr[1])[0].tolist()
    assert len(medial_wall) == n_medial[hemi]
    assert len(medial_wall) + len(cortical_vertices) == n_vertices

    for i, ds in enumerate([ds1, ds2, ds3, ds4]):
        print(i)
        ds.samples = ds.samples[]
        if i < 2:
            ses = 0
        else:
            ses = 1
        if zscore_features:
            mv.zscore(ds, chunks_attr=None)
        # ds.sa.pop('stats')
        # ds.a.pop('history')
        # ds.sa['conditions'] = [c[:-7] for c in ds.sa.labels]
        # ds.sa['targets'] = ds.sa.conditions
        ds.sa['sessions'] = [ses] * ds.shape[0]

        ds.fa['node_indices'] = range(ds.shape[1])
        ds.fa['center_ids'] = range(ds.shape[1])
        ds.sa['targets'] = range(ds.shape[0])
        # ds.sa.pop('labels')

        if hyperalign:
            ds = mappers[i][participant].forward(ds)
            print("Hyperaligned participant {0}".format(participant))
            if zscore_features:
                mv.zscore(ds, chunks_attr=None)
            ds.fa['node_indices'] = range(ds.shape[1])
            ds.fa['center_ids'] = range(ds.shape[1])

    ds_all = mv.vstack((ds1, ds2, ds3, ds4), fa='update')
    rsa.PDist(**kwargs)
    #variant_ids = mv.remove_invariant_features(ds_both).fa.center_ids.tolist()

    # Set up cross-validated RSA
    cv_rsa_ = mv.CrossValidation(mv.CDist(pairwise_metric='correlation'),
                                 mv.HalfPartitioner(attr='sessions'),
                                 errorfx=None)

    # cv_rsa above would return all kinds of .sa which are important
    # but must be the same across searchlights. so we first apply it
    # to the entire ds to capture them
    cv_rsa_out = cv_rsa_(ds_all)
    target_sa = cv_rsa_out.sa.copy(deep=True)

    # And now create a postproc which would verify and strip them off
    # to just return samples
    from mvpa2.testing.tools import assert_collections_equal
    from mvpa2.base.collections import SampleAttributesCollection
    from mvpa2.base.node import Node
    def lean_errorfx(ds):#Node):
        #def __call__(self, ds):
            assert_collections_equal(ds.sa, target_sa)
            # since equal, we could just replace with a blank one
            ds.sa = SampleAttributesCollection()
            return ds
    # the one with the lean one
    cv_rsa = mv.CrossValidation(mv.CDist(pairwise_metric='correlation'),
                                 mv.HalfPartitioner(attr='sessions'),
                                 errorfx=None, postproc=lean_errorfx)

    sl = mv.Searchlight(cv_rsa, queryengine=qe, enable_ca=['roi_sizes'],
                        nproc=1, results_backend='native')
    #sl = mv.Searchlight(cv_rsa, queryengine=qe, enable_ca=['roi_sizes'],
    #                    nproc=1, results_backend='native', roi_ids=cortical_vertices)
    #tmp_prefix='/local/tmp/sam_sl_p{0}_{1}_'.format(participant_id, hemi)
    mv.debug.active += ['SLC']
    sl_result = sl(ds)
    assert len(sl_result.sa) == 0  # we didn't pass any
    sl_result.sa = target_sa

    print '>>>', np.mean(sl.ca.roi_sizes), np.std(sl.ca.roi_sizes)

    sl_means = np.mean(np.dstack((sl_result.samples[:n_conditions**2, :],
                                  sl_result.samples[n_conditions**2:, :])),
                       axis=2)
    sl_final = mv.Dataset(
         sl_means,
         sa={'conditions': sl_result.sa.conditions[:sl_means.shape[0], :].tolist(),
             'participants': [int(participant[-2:])] * sl_means.shape[0]},
         fa=sl_result.fa, a=sl_result.a)
    #assert sl_result.shape[0] == n_conditions**2
    print(sl_final)
    mv.h5save('/idata/DBIC/cara/life/search_RDMs_sq_zscore_HA_{0}_{1}.hdf5'.format(participant, hemi), sl_final)
        #mv.niml.write(join(mvpa_dir, 'search_RDMs_sq_p{0}_{1}_TEST.niml.dset'.format(
        #                                       participant_id, hemi)), sl_result)
