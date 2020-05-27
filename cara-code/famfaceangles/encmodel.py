"""Module containing functions for Forward-Encoding Models"""
import numpy as np
from joblib.parallel import Parallel, delayed
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
import types


# from https://stackoverflow.com/questions/25191620/
# creating-lowpass-filter-in-scipy-understanding-methods-and-units
def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=6):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, method='gust')
    return y


def resample(array, origfreq=30.0, sampfreq=1.0):
    """
    Resample data to match TR frequency. First data is low-passed to sampfreq
    Hz, then it is decimated to match the TR frequency.

    Parameters
    ----------
    array : array-like
        the data to resample
    origfreq : float
        original sampling frequency (Hz)
    sampfreq : float
        new sampling frequency

    Returns
    -------
    resampled : array-like
        the resampled data
    """

    resampled = butter_lowpass_filter(array, sampfreq, origfreq)
    select = slice(None, None, int(origfreq/sampfreq))
    return resampled[select]


def _conform_input(video):
    """Return an array regardless of the input. Will have to store all data
    in memory"""
    video_ = [np.asarray(v, dtype=np.float) for v in video]
    video_ = np.array(video_)
    if video_.ndim == 4:
        video_ = video_.mean(axis=-1)
    # normalize by the mean luminance for each frame
    video_ -= video_.mean(axis=(1, 2))[:, None, None]
    return video_


def global_motion(video, prev_frame=None, max_frames=None):
    """
    Compute global motion estimate for a video. For two consecutive frames,
    global motion is computed as the average difference between the two
    frames across all channels. Each frame is normalized by the average
    luminance to avoid counting changes in luminance as motion.

    Parameters
    ----------
    video : array or iterable
        contains frames of the video. If array, the first dimension needs
        to be time. Otherwise a list of frames, or an imageio.Reader
        object can be passed.
    prev_frame : array (n_x, n_y, n_channels) or None
        previous frame used to compute the global motion for the first frame
        of the passed video. This can be used to stitch together a series of
        shorter clips.
    max_frames : int or None
        max number of frames required. Use this to make sure that the output
        has length exactly `max_frames`. If the video length is less than
        `max_frames`, the last motion estimate will be repeated to match the
        desired length.

    Returns
    -------
    motion : array-like (max_frames, )
        the global motion estimate
    """
    video = _conform_input(video)
    prev = None
    if prev_frame is not None:
        prev = prev_frame.copy()
        # preprocess prev as _conform_input
        prev = _conform_input([prev])[0]
    n_frames = len(video)
    max_frames = n_frames if max_frames is None else max_frames
    extra_frames = max_frames - n_frames
    video = video[:max_frames]
    motion = np.sqrt(
        np.mean((video[1:] - video[:-1])**2, axis=(1, 2))).tolist()
    first = 0. if prev is None else np.sqrt(np.mean((video[0] - prev)**2))
    motion = [first] + motion + [motion[-1]] * extra_frames
    return np.asarray(motion)


def add_delays(X, delays=(2, 4, 6), tr=1):
    """
    Given a design matrix X, add additional columns for delayed responses.

    Parameters
    ----------
    X : array (n_samples, n_predictors)
    delays : array-like (n_delays, )
        delays in seconds
    tr : float

    Returns
    -------
    Xd : array (n_samples, n_predictors * n_delays)
        design matrix with additional predictors
    """
    s, p = X.shape
    d = len(delays)
    Xd = np.zeros((s, p * (d + 1)))
    Xd[:, :p] = X
    for i, delay in enumerate(delays, 2):
        start = int(delay * tr)
        Xd[start:, (i - 1) * p:i * p] = X[:s - start]
    return Xd


def zscore(x):
    """x is (samples, features). zscores each sample individually"""
    axis = None if x.ndim == 1 else 1
    return (x - x.mean(axis=axis)[:, None])/x.std(axis=axis, ddof=1)[:, None]


def corr(x, y):
    """Return the correlation between the pairwise rows of x and y"""
    nf = len(x) if x.ndim == 1 else x.shape[1]
    axis = None if x.ndim == 1 else 1
    x_ = zscore(x)
    y_ = zscore(y)
    r = (x_ * y_).sum(axis=axis)/(nf-1)
    return r


def _ridge_search(X, Y, train, test, alphas=np.logspace(0, 4, 20),
                  scoring=corr):
    """Fit ridge regression sweeping through alphas

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    train : array
        index array for training
    test : array
        index array for testing
    alphas : array
        alpha values used to fit
    scoring : callable
        function used to score the fit (default correlation)

    Returns
    -------
    scores : array (n_alphas, n_features)
        score of prediction for each alpha and each feature
    """
    ridge = Ridge(fit_intercept=False, solver='svd')
    scores = []
    for alpha in alphas:
        ridge.set_params(alpha=alpha)
        ridge.fit(X[train], Y[train])
        scores.append(scoring(Y[test].T, ridge.predict(X[test]).T))
    return np.array(scores)


def ridge_search(X, Y, cv, alphas=np.logspace(0, 4, 20), scoring=corr,
                 njobs=1):
    """Fit ridge regression sweeping through alphas across all
    cross-validation folds

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    cv : iterable or generator
        returning (train, test) tuples with indices for row of X and Y
    alphas : array
        alpha values used to fit
    scoring : callable
        function used to score the fit (default correlation)
    njobs : int
        number of parallel jobs to run. Each cross-validation fold will be
        run in parallel

    Returns
    -------
    scores : array (n_folds, n_alphas, n_features)
        score of prediction for each alpha and each feature
    """
    scores = Parallel(n_jobs=njobs)(
        delayed(_ridge_search)(X, Y, train, test, alphas=alphas,
                               scoring=scoring)
        for train, test in cv)
    return np.array(scores)


def ridge_optim(X, Y, group, train, test, alphas, njobs=1, nblocks=1):
    """Main loop for nested cross-validation. It will perform parameter search
    within the training set, and then return the best alpha and the score
    (correlation). Parameter search is performed by averaging the prediction
    score curves across features and folds, and finding the optimal global
    alpha (across features and folds).

    Note that X and Y are assumed to be centered (e.g., z-scored), because
    the intercept is not fitted.

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    group : array (n_samples,)
        indicator variable used for grouping the sample (e.g., runs or chunks),
        cross-validation within training set will be performed according to
        this grouping factor using LeaveOneGroupOut.
    train : array
        indices for training set
    test : array
        indices for testing set
    alphas : array
        alphas to search through for optimal search
    njobs : int
        number of parallel jobs
    nblocks : int
        number of blocks for parallel jobs (more use less memory)

    Returns
    -------
    score : array (n_features)
        score of optimal prediction for each feature
    weights : array (n_features, n_predictors)
        weights of optimal estimator
    best_alpha : float
        global best alpha used to obtain score
    score_curve : array (n_alphas)
        global score curve
    """
    # Split into training, testing
    y_tr, y_te = Y[train], Y[test]
    x_tr, x_te = X[train], X[test]
    # grouping factor, e.g., runs
    gr_tr, gr_te = group[train], group[test]
    n_features = Y.shape[1]
    n_blocks = max(njobs, nblocks)
    blocks = np.array_split(np.arange(n_features), n_blocks)
    # Work on training, find best alpha by cross-validation
    # this is parallelized across blocks to reduce memory consumption
    # score_tr is (n_folds, n_alphas, n_features)
    cv = list(LeaveOneGroupOut().split(x_tr, groups=gr_tr))
    score_tr = Parallel(n_jobs=njobs)(
        delayed(ridge_search)(x_tr, y_tr[:, ib], cv, alphas=alphas) for ib in
        blocks)
    score_tr = np.dstack(score_tr)
    score_curve = score_tr.mean(axis=(0, 2))
    # because this is correlation, we need to find the max
    best_alpha = alphas[np.argmax(score_curve)]

    # Fit on training, predict on testing, store prediction for each voxel
    ridge = Ridge(alpha=best_alpha, fit_intercept=False, solver='svd')
    ridge.fit(x_tr, y_tr)
    final_score = corr(y_te.T, ridge.predict(x_te).T)

    return final_score, ridge.coef_, best_alpha, score_curve


def fit_encmodel(X, Y, group, alphas, njobs=1, nblocks=1):
    """
    Fit encoding model using features in X to predict data in Y, testing a
    range of different alphas.

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    group : array (n_samples,)
        indicator variable used for grouping the sample (e.g., runs or chunks),
        cross-validation will be performed according to this grouping factor
        using LeaveOneGroupOut.
    alphas : array
        alphas to search through for optimal search
    njobs : int
        number of parallel jobs
    nblocks : int
        number of blocks for parallel jobs (more use less memory)

    Returns
    -------
    score : array (n_folds, n_features)
        score of optimal prediction for each feature
    weights : array (n_folds, n_features, n_predictors)
        weights of optimal estimator
    best_alpha : array (n_folds, )
        global best alpha used to obtain score
    score_curve : array (n_folds, n_alphas)
        global score curves

    """
    logo = LeaveOneGroupOut()
    cv = logo.split(X, groups=group)

    out = []
    for i, (train, test) in enumerate(cv):
        print("Running CV {}".format(i))
        out.append(ridge_optim(X, Y, group, train, test,
                               alphas=alphas, njobs=njobs, nblocks=nblocks))
    out = zip(*out)
    return [np.array(o) for o in out]
