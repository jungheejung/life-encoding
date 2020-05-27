import matplotlib; matplotlib.use('agg')
from scipy.spatial.distance import pdist
import mvpa2.suite as mv
from mvpa2.measures import rsa
from mvpa2.mappers.fx import mean_group_feature
from mvpa2.measures.searchlight import sphere_searchlight


import numpy as np
from os.path import join

hyperalign = True if int(sys.argv[1]) == 0 else False
print("hyperalign?", hyperalign)

sam_data_dir = '/idata/DBIC/snastase/life'
suma_dir = '/idata/DBIC/snastase/life/SUMA'

tr = {1:374, 2:346, 3:377, 4:412}
n_medial = {'lh': 3486, 'rh': 3491}
n_vertices = 40962

runs = range(1,5)
participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']
hemispheres = ['lh', 'rh']

list_of_RDMs = []

for hemi in hemispheres:
    # Load surface and create searchlight query engine
    surf = mv.surf.read(join(suma_dir, '{0}.pial.gii'.format(hemi)))
    qe = mv.SurfaceQueryEngine(surf, 20.0, distance_metric='dijkstra')
    print("Finished creating surface-based searchlight query engine")

    rdm = mv.PDist(pairwise_metric='correlation', center_data=False)

    sl = mv.Searchlight(rdm, queryengine=qe, enable_ca=['roi_sizes'],
                        nproc=6, roi_ids=cortical_vertices)
    mv.debug.active += ['SLC']
    for run in runs:
        if hyperalign:
            mappers = mv.h5load(join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(hemi, run)))
        for participant in participants:

            # print(ds.sa['intents'])
            # to_avg = mean_group_feature()
            # averaged_response = to_avg(ds)

            print("loading data for run {0}, hemisphere {1}, participant {2}...".format(run, hemi, participant))
            ds = mv.gifti_dataset(join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[run], run, hemi)))
            mv.zscore(ds, chunks_attr=None)

            if hyperalign:
                ds = mappers[participant].forward(ds)
                mv.zscore(ds, chunks_attr=None)
            ds.fa['node_indices'] = range(ds.shape[1])


            # n_samples = ds.samples.shape[0]
            #
            # # Exclude medial wall
            # print(np.where(np.sum(ds.samples == 0, axis=0) == n_samples))
            n_samples = ds.samples.shape[0]
            medial_wall = np.where(np.sum(ds.samples == 0, axis=0) == n_samples)[0].tolist()
            print(len(medial_wall))
            cortical_vertices = np.where(np.sum(ds.samples == 0, axis=0) < n_samples)[0].tolist()
            assert len(medial_wall) == n_medial[hemi]
            assert len(medial_wall) + len(cortical_vertices) == n_vertices

            sl_result = sl(ds)
            print(ds.samples.shape, sl_result.samples.shape)
            list_of_RDMs.append(sl_result)
        final = mv.vstack(list_of_RDMs)
        print(final.shape)
        mv.h5save('/idata/DBIC/cara/search_hyper_mappers_life_mask_nofsel_{0}_{1}_leftout_{1}_{2}.hdf5'.format(participant, hemi, left_out, sys.argv[1]), final)
