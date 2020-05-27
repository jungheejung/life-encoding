import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import os

# nlabels = {'all':, 'bg':, 'animals':, 'actions':}
# semantic_dir = '/idata/DBIC/cara/life/semantic_cat/'
# for model in nlabels.keys():
#     data = np.load(os.path.join(semantic_dir, '{0}.npy'.format(model)))
#     pca = decomposition.PCA(n_components=nlabels[model]-1)
#     pca.fit(data)
#     pca_data = pca.transform(data)
#     print(pca_data.shape)
#     np.save(os.path.join(semantic_dir, '{0}_pca.npy'.format(model)), pca_data)


for f in os.listdir('/dartfs-hpc/scratch/cara/w2v/semantic_categories/'):
    l = pd.read_csv(os.path.join('/dartfs-hpc/scratch/cara/w2v/semantic_categories/', f))
    print(f, l.shape)
