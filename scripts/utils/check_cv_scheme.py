# purpose:
# see if there's a sklearn object that constructs the same loro scheme
# loro1 = [(train_id[:dur1 + dur2], train_id[dur1 + dur2:]),
        #  (np.concatenate((train_id[:dur1], train_id[dur1 + dur2:]), axis=0),
        #   train_id[dur1:dur1 + dur2]),
        #  (train_id[dur1:], train_id[:dur1])]
# train_id = np.arange(X1train.shape[0])
# dur1, dur2, dur3 = tr_movie[included[0]] - \
#     3, tr_movie[included[1]] - 3, tr_movie[included[2]] - 3

# %%libraries
import numpy as np
from sklearn.model_selection import PredefinedSplit

# %% https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html#sklearn.model_selection.PredefinedSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
test_fold = [0, 1, -1, 1]
ps = PredefinedSplit(test_fold)
# %%
ps.get_n_splits()

for train_index, test_index in ps.split():
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
# %%
