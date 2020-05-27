import matplotlib; matplotlib.use('agg')

import numpy as np
from os.path import join
import matplotlib.pyplot as plt


data_dir = '/dartfs-hpc/scratch/cara/'
models = ['ws', 'aa', 'ha_common', 'ha_testsubj']
runs = range(1,5)
hemispheres= ['lh', 'rh']
participants = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']
#
# # l = []
# # for m in models:
# #     print(m)
# #     for r in runs:
# #         for p in participants:
# #             l.append(np.load(join(data_dir, '{0}-leftout{1}'.format(m, r), p, 'lh/alphas.npy')))
# # alphas = np.concatenate(l)
# # print(alphas.shape)
#
# from sklearn.linear_model import RidgeCV
#
#
# def get_stim_for_fold(fold_shifted, included):
#     cam = np.load('/idata/DBIC/cara/life/semantic_cat/all.npy')
#     full_stim = []
#     full_stim.append(cam[:366,:])
#     full_stim.append(cam[366:704,:])
#     full_stim.append(cam[704:1073,:])
#     full_stim.append(cam[1073:,:])
#
#     for i in range(len(full_stim)):
#         this = full_stim[i]
#         full_stim[i] = np.concatenate((this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)
#         print(i+1, full_stim[i].shape)
#
#     train_stim = [full_stim[i] for i in np.subtract(included, 1)]
#     test_stim = full_stim[fold_shifted-1]
#
#     print(len(train_stim), test_stim.shape)
#     return np.vstack(train_stim), test_stim
#
# # Load the data
# def get_ha_common_data(test_p, mappers, fold_shifted, included):
#     hemi = 'lh'
#     train_p = [x for x in participants if x != test_p]
#     print('\nLoading fMRI GIFTI data for HA in common space and using {0} as test participant...'.format(test_p))
#     train_resp = []
#     for run in included:
#         avg = []
#         for participant in train_p:
#             if run == 4:
#                 resp = mappers[participant].forward(mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[run], run, hemi))).samples[4:-14,:])
#             else:
#                 resp = mappers[participant].forward(mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr[run], run, hemi))).samples[4:-7,:])
#
#             # z-score features across samples
#             mv.zscore(resp, chunks_attr=None)
#             avg.append(resp)
#
#         avg = np.mean(avg, axis=0)
#         print(run, avg.shape)
#         train_resp.append(avg)
#
#     if fold_shifted == 4:
#         test_resp = mappers[participant].forward(mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr[fold_shifted], fold_shifted, hemi))).samples[4:-14,:])
#     else:
#         test_resp = mappers[participant].forward(mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr[fold_shifted], fold_shifted, hemi))).samples[4:-7,:])
#     # z-score features across samples
#     mv.zscore(test_resp, chunks_attr=None)
#
#     return train_resp, test_resp
#
# train_stim, test_stim = get_stim_for_fold(3, [1,2,4])
#
#
# # if ha_common or ha_testsubj:
# #     print('\nLoading hyperaligned mappers...')
# #     mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(hemi, 3)))
#
# train_resp, test_resp = get_ha_common_data('sub-rid000012', mappers, 3, [1,2,4])

# p = 'sub-rid000006'
h = 'lh'
model = 'aa'

alphas = np.logspace(-3,3,100)

for space in ['smallspace']:
    for model in ['aa', 'ws', 'ha_common', 'ha_testsubj']:
        for run in range(1,5):
            for p in participants:
                if space == 'smallspace':
                    alphas = np.logspace(0,2,20)
                    lh_boot_corrs = np.load(join(data_dir, 'models', '{0}-leftout{1}_feweralphas'.format(model, run), p, 'lh', 'bootstrap_corrs.npy'))
                    rh_boot_corrs = np.load(join(data_dir, 'models', '{0}-leftout{1}_feweralphas'.format(model, run), p, 'rh', 'bootstrap_corrs.npy'))
                else:
                    alphas = np.logspace(-3,3,100)
                    boot_corrs = np.load(join(data_dir, 'three_boots', '{0}-leftout{1}'.format(model, run), p, h, 'bootstrap_corrs.npy'))
                print(lh_boot_corrs.shape)

                lh = np.mean(lh_boot_corrs,axis=2)
                rh = np.mean(rh_boot_corrs,axis=2)
                total = np.concatenate((lh, rh), axis=1)
                print(total.shape)
                np.save('alpha_corrs/alphas_{0}_{1}_run{2}_{3}.npy'.format(model, p, run, space), np.mean(total,axis=1))
                print(np.mean(total,axis=1).shape)
                # plt.plot(alphas, np.mean(total,axis=1))
                # plt.savefig('plots/alphas_{0}_{1}_run{2}_{3}.png'.format(model, p, run, space))
                # plt.close()


# # Create a list of alphas to cross-validate against
# l = [10, 20, 30]
# for num in l:
#     alphas = np.logspace(0, 2, num)
#     # Instantiate the linear model and visualizer
#     model = RidgeCV(alphas=alphas, store_cv_values=True)
#
#     print(num, X_train.shape, y_train.shape)
#     model.fit(X_train, y_train)  # Fit the training data to the visualizer
#     pred = model.predict(X_test)
#     print(model.cv_values_.shape)
#     mse = np.mean(model.cv_values_, axis=0)
#     print(mse.shape)
#     mse = np.mean(mse, axis=0)
#     print(mse.shape)
#
#     plt.plot(alphas, mse)
#     plt.savefig('alphas_{0}_sub12_02.png'.format(num))
#
#     nnpred = np.nan_to_num(pred)
#     corrs = np.nan_to_num(np.array([np.corrcoef(y_test[:,ii], nnpred[:,ii].ravel())[0,1] for ii in range(y_test.shape[1])]))
#     mv.niml.write('check{0}_sub12_02.lh.niml.dset'.format(num), corrs[None,:])
#     plt.close()
