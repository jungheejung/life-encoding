# %% load libraries
import sklearn
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
import numpy as np
# %%
# fold = 2
fold_shifted = 4
included = [1, 2, 3, 4]
included.remove(fold_shifted)
tr_movie = {1: 369, 2: 341, 3: 372, 4: 406}
tr_fmri = {1: 374, 2: 346, 3: 377, 4: 412}
# TODO: match movie and fmri from the getgo. 
# don't do it within CV scheme
# %%

X1train = np.zeros((1073, 120))
Ytrain = np.zeros((1073, 3))
train_id = np.arange(X1train.shape[0])
dur1, dur2, dur3 = tr_movie[included[0]] - \
    3, tr_movie[included[1]] - 3, tr_movie[included[2]] - 3
Ytrain_id = np.arange(Ytrain.shape[0])

# %% Sam 01/25/2022
# run_id 
tr_movie= {1: 366, 2: 338, 3: 369, 4: 403}
run_tr = np.concatenate([ [r] *t for r, t in tr_movie.items()])
print(run_tr)
loro_outer = sklearn.model_selection.PredefinedSplit(run_tr)
loro_outer.get_n_splits(np.arange(1476), np.arange(1476), run_tr)

# %%
X = np.arange(1476)
y = np.arange(1476)

for train_index, test_index in loro_outer.split(X, y): # OUTER LOOP
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# INNER
    loro_inner = sklearn.model_selection.PredefinedSplit(run_tr[train_index])

    model = GroupRidgeCV(groups="input",
    cv=loro_inner,
    fit_intercept=False, # TODO: figure out fit_intercept
    solver_params=dict(score_func=correlation_score,
       progress_bar=True,
    n_iter=1000))
    model.fit([X0_train[train_index],X1_train[train_index],
               X_train[train_index]],Y_train[train_index]) 
    # https://gallantlab.github.io/himalaya/_generated/himalaya.ridge.GroupRidgeCV.html#himalaya.ridge.GroupRidgeCV
    print("TRAIN:", train_index, "TEST:", test_index)
    # break
    print(X_train, X_test, y_train, y_test)


    Y_pred=model.predict([X0_test,X1_test,X_test],split=True)[2]