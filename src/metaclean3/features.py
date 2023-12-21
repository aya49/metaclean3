import numpy as np
import pandas as pd
import warnings

from scipy.spatial import cKDTree
from scipy.stats import moment
from sklearn.preprocessing import MinMaxScaler

# detect if GPU is available to calculate density
try:
    import torch
    G_RAP_GPU = torch.cuda.is_available()
    if G_RAP_GPU:
        import cuml
except ModuleNotFoundError:
    G_RAP_GPU = False

pd.set_option('mode.chained_assignment', None)

def scale_values(data: pd.DataFrame):# -> pd.DataFrame:
    """Scales each column of the given `pandas.DataFrame` to range [0, 1).

    Args:
        data (pandas.DataFrame): A numeric (int/float) matrix.

    Returns:
        pandas.DataFrame: The scaled version of the given matrix.
    """
    ds = data.apply(lambda x: np.squeeze(np.abs(MinMaxScaler().fit_transform(
        np.array(x - 1).reshape((-1, 1)) ))), axis=0)
    return ds

## feature functions ####
def get_bin_density(
    data: np.ndarray | pd.DataFrame,
    bins: list | np.ndarray | pd.Series,
    dens_agg_type: str = 'max',
    n_cores: int = -1,
    p: int = 2, # see kdtree.query
    eps: float = 0.1, # see kdtree.query
    dens_k_dtm: int = 15
):# -> np.ndarray:
    """Calculates a density value for each bin.

    Args:
        data (numpy.ndarray | pandas.DataFrame): Numeric matrix (2D array).
        bins (list | numpy.ndarray | pandas.Series):
            list containing bin value for each `data` row.
        dens_agg_type (str, optional): Aggregation method per bin.
            See pandas.DataFrame.agg(). Defaults to 'max'.
        n_cores (int, optional): The number of cores to use while
            calculating the density feature. Set to -1 to use all cores.
            Defaults to -1.
        p (int, optional): See scipy.spatial.cKDTree.query(). Defaults to 2.
        eps (float, optional): See scipy.spatial.cKDTree.query().
            Defaults to 0.1.
        dens_k_dtm (int, optional): See scipy.spatial.cKDTree.query().
            Defaults to 15.

    Returns:
        numpy.ndarray: A 1D array containing a density value for each bin.
    """
    if len(bins) != len(data):
        raise ValueError('bins and data do not have the same length.')

    # estimate density for each bin
    if G_RAP_GPU:
        model = cuml.neighbors.NearestNeighbors(n_neighbors=dens_k_dtm, p=2)
        model = model.fit(data)
        dd, _ = model.kneighbors(data)
    else:
        kdtree = cKDTree(data)
        dd, _ = kdtree.query(data, k=dens_k_dtm, p=p, workers=n_cores, eps=eps)

    de = (dd ** data.shape[1]).sum(-1) # TODO: comment
    with warnings.catch_warnings():
        # known divide by zero warning, rare
        warnings.simplefilter("ignore")
        d = -np.log(de)
    d[(de == 0) | (de < 0)] = 0
    d[np.isinf(de)] = np.max(d[~np.isinf(d)])
    d = np.array(pd.DataFrame({'d': d, 'bin': bins}).groupby(
        'bin', as_index=False)['d'].agg(dens_agg_type)['d'])

    return d

def get_bin_moments(
    data: np.ndarray | pd.DataFrame,
    bins: list | np.ndarray | pd.Series,
    mmt: int = 2
):# -> np.ndarray:
    """Calculates the `mmt`th moment of each bin.

    Args:
        data (numpy.ndarray | pandas.DataFrame): Numeric matrix (2D array).
            If data has more than one column, we take the sum of each row.
        bins (list | np.ndarray | pd.Series):
            list containing bin value for each `data` row.
        mmt (int, optional): Moment. Defaults to 2.

    Returns:
        numpy.ndarray: A 1D array containing the `mmt`th moment for each bin.
    """
    # estimate variance/skew for each bin
    data = data if data.shape[1] == 1 else np.sum(data, axis=1)
    _, inds = np.unique(bins, return_index=True)
    dv = np.squeeze(np.array(data))
    bsp = np.array(np.split(dv, inds)[1:], dtype=object)
    if len(bsp.shape) == 2: # i.e. all bins same size; work around
        dv = np.concatenate((dv, np.array([0.0], dtype=np.dtype(dv[0]))))
        bsp = np.array(np.split(dv, inds)[1:], dtype=object)
        bsp[-1] = bsp[-1][:-1]
    with warnings.catch_warnings():
        # known catastrophic cancellation warning; values replaced
        warnings.simplefilter("ignore")
        v = np.vectorize(lambda x: np.sum(moment(x, moment=mmt, axis=0)))(bsp)

    # sum and remove infinite values
    not_inf = ~np.isinf(v)
    if np.sum(not_inf) < len(v):
        v[v == np.Infinity] = np.max(v[not_inf])
        v[v == -np.Infinity] = np.min(v[not_inf])

    return v


