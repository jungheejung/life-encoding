# 1) get medial vs. non-medial indices
nonmedial = np.where(cortical_vertices[hemi] == 1)[0]
medial = np.where(cortical_vertices[hemi] == 0)[0]

# 2) get intersection of ROI vertices and medial/non-medial indices
selected_node = np.intersect1d(nonmedial, np.array(ROI_dict[hemi][roi]))
medial_node = np.intersect1d(medial, np.array(ROI_dict[hemi][roi]))

# 3) one example of dataloader, with "selected node"
def get_ws_data(test_p, fold_shifted, included, hemi, selected_node):
    print("4. within subject data")
    print(
        'Loading fMRI GIFTI data for HA in test subj space and using {0} as test participant...'.format(test_p))
    train_resp = []
    for run in included:
        avg = []
        if run == 4:
            resp = mv.gifti_dataset(os.path.join(
                sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[run], run, hemi))).samples[4:-5, :]
        else:
            resp = mv.gifti_dataset(os.path.join(
                sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_p, tr_fmri[run], run, hemi))).samples[4:-4, :]

        resp = resp[:, selected_node]

        mv.zscore(resp, chunks_attr=None)
        print('train', run, resp.shape)
        train_resp.append(resp)

    if fold_shifted == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-5, :]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(
            test_p, tr_fmri[fold_shifted], fold_shifted, hemi))).samples[4:-4, :]

    # older code, now updated with selected noce
    # test_resp = test_resp[:, cortical_vertice[hemi] == 1]
    test_resp = test_resp[:, selected_node]
    mv.zscore(test_resp, chunks_attr=None)
    print('test', fold_shifted, test_resp.shape)

    return train_resp, test_resp


# 4) convert node indices into numpy arrays
ind_nonmedial = np.array(selected_node) # insert nonmedial index
ind_medial = np.array(medial_node) # insert medial index
append_zero = np.zeros(len(medial_node)) # insert medial = 0
alpha_nonmedial = np.array(new_alphas) # insert nonmedial alpha

# 5) concatenate medial & non-medial indices
#    concatenate alpha values from non-medial nodes and zero vales (if medial nodes exist.)
if len(medial_node) != 0:
    index_chunk = np.concatenate((ind_nonmedial,ind_medial), axis = None)
    alpha_value = np.concatenate((alpha_nonmedial,append_zero),axis = None)
else:
    index_chunk = ind_nonmedial
    alpha_value = alpha_nonmedial

# 6) zip medial & non-medial index and alpha values - sort. 
zipped_alphas = zip(index_chunk.astype(float), alpha_value.astype(float))
sorted_alphas = sorted(zipped_alphas)

# 6-4. save alpha
alpha_savename = os.path.join(directory, 'hyperparam-alpha_{0}_model-{1}_align-{2}_foldshifted-{3}_hemi-{4}_range-{5}.json'.format(
        test_p, model, align,  fold_shifted, hemi,roi))
with open(alpha_savename, 'w') as f:
     json.dump(sorted_alphas, f)