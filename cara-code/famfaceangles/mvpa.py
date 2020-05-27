"""Module containing various functions and classes for MVPA"""
import numpy as np
from mvpa2.measures.base import Measure
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import EnsureChoice
from mvpa2.datasets.base import Dataset
from scipy import stats, linalg
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize as sknorm


def pearsonr_no_pval(x, y):
    """Returns pearsons correlation without pvalues"""
    return stats.pearsonr(x, y)[0]


def spearmanr_no_pval(x, y):
    """Returns spearman correlation without pvalues"""
    return stats.pearsonr(stats.rankdata(x), stats.rankdata(y))[0]


def pcorr(y, X, corrfx=pearsonr_no_pval, normalize=True):
    """
    Compute partial correlation between y and columns of X. It returns
    Parameters
    ----------
    y : np.array (n_samples, 1)
        target vector (e.g., neural dissimilarity matrix)
    X : np.array (n_samples, p)
        vectors that will be correlated with y, partialing out the effect
        of the other columns (e.g., behavioral RDMs)
    corrfx : function (x, y)
        function to apply on the residuals to compute the correlation;
        default is pearson r, but spearman r could be used as well.
    normalize : boolean
        whether to normalize (demean and unit norm) both y and X. Otherwise
        you'll have to input an additional columns of ones to account for
        differences in means.

    Returns
    -------
    rp : np.array (p, )
        pairwise partial correlations between y and the columns of X
    """
    # This function was inspired by the code by Fabian Pedregosa, available
    # here: https://gist.github.com/fabianp/9396204419c7b638d38f

    X = np.asarray(X)
    y = np.asarray(y)
    n_corr = X.shape[1]
    if n_corr < 2:
        raise ValueError("Need more than one column in X to run partial corr")
    if normalize:
        X = X - X.mean(axis=0)
        y = y - y.mean(axis=0)
        X = sknorm(X, axis=0)
        y = sknorm(y, axis=0)
    y = y.flatten()

    rp = np.zeros(n_corr, dtype=np.float)
    for i in range(n_corr):
        idx = np.ones(n_corr, dtype=np.bool)
        idx[i] = False
        beta_y = linalg.lstsq(X[:, idx], y)[0]
        beta_i = linalg.lstsq(X[:, idx], X[:, i])[0]

        res_y = y - X[:, idx].dot(beta_y)
        res_i = X[:, i] - X[:, idx].dot(beta_i)

        rp[i] = corrfx(res_y, res_i)
    return rp


CORRFX = {
    'pearson': pearsonr_no_pval,
    'spearman': spearmanr_no_pval
}


class PCorrTargetSimilarity(Measure):
    """Calculate the partial correlations of a neural RDM with more than two
    target RDMs. This measure can be used for example when comparing a
    neural RDM with more than one behavioral RDM, and one desires to look
    only at the correlations of the residuals.

    NOTA BENE: this measure computes a distance internally through
    scipy.spatial.pdist, thus you should make sure that the
    predictors are in the correct direction, that is smaller values imply
    higher similarity!
    """

    is_trained = True
    """Indicate that this measure is always trained."""

    pairwise_metric = Parameter('correlation', constraints='str', doc="""\
          Distance metric to use for calculating pairwise vector distances for
          the neural dissimilarity matrix (DSM).  See
          scipy.spatial.distance.pdist for all possible metrics.""")

    correlation_type = Parameter('spearman', constraints=EnsureChoice(
        'spearman', 'pearson'), doc="""\
          Type of correlation to use between the compute neural RDM and the
          target RDMs. If spearman, the residuals are ranked
          prior to the correlation.""")

    normalize_rdms = Parameter(True, constraints='bool', doc="""\
          If True then center and normalize each column of the neural RDM
          and of the predictor RDMs by subtracting the
          column mean from each element and imposing unit L2 norm.""")

    def __init__(self, target_rdms, **kwargs):
        """
        Parameters
        ----------
        target_rdms : array (length N*(N-1)/2, n_predictors)
          Target dissimilarity matrices
        """
        # init base classes first
        super(PCorrTargetSimilarity, self).__init__(**kwargs)
        self.target_rdms = target_rdms
        self.corrfx = CORRFX[self.params.correlation_type]
        self.normalize_rdms = self.params.normalize_rdms

    def _call(self, dataset):
        """
        Parameters
        ----------
        dataset : input dataset

        Returns
        -------
        Dataset
        each sample `i` correspond to the partial correlation between the
        neural RDM and the `target_dsms[:, i]` partialling out
        `target_dsms[:, j]` with `j != i`.
        """
        data = dataset.samples
        dsm = pdist(data, self.params.pairwise_metric)
        rp = pcorr(dsm[:, None],
                   self.target_rdms,
                   corrfx=self.corrfx,
                   normalize=self.normalize_rdms)
        return Dataset(rp[:, None])
