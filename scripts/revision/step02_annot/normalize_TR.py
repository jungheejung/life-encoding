# ses-03_run-03_order-01_content-normativeprosocial2
# 'ses-03_run-03_order-01_content-normativeprosocial2':230,
# %%
length = 230
# library
import numpy as np
# 
for feature in ['actions', 'agents', 'objects', 'scenes' ]:
    fname = f'/Users/h/Downloads/life/ses-03_run-03_order-01_content-normativeprosocial2_feature-{feature}.npy'
    data = np.load(fname)
    new_data = data[:-1]
    print(new_data.shape)
    assert length == new_data.shape[0]
    np.save(fname, new_data)
# %%
