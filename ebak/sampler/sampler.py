# Third-party
import numpy as np
from scipy.interpolate import interp1d

__all__ = ['design_matrix', 'sinusoid_model', 'period_grid',
           'tensor_vector_scalar', 'marginal_ln_likelihood',
           'sample_posterior']

def design_matrix(t, P, n_terms=1):
    """

    Parameters
    ----------
    t : array_like [day]
        Array of times in days.
    P : numeric [day]
        Value of the period in days.
    n_terms : int (optional)
        Number of terms in the Fourier series.
    """
    t = np.atleast_1d(t)
    a = np.ones_like(t)

    # design matrix
    X = a.reshape(1,-1)
    for k in range(1,n_terms+1):
        x_k = np.cos(2*np.pi*k*t / P)
        y_k = np.sin(2*np.pi*k*t / P)
        X = np.vstack((X, x_k, y_k))
    return X.T

def sinusoid_model(p, t, P, n_terms=1):
    """

    Parameters
    ----------
    p : array_like
        Parameter vector. For a 1-term fit, this is `(v0, a1, b1)`.
    t : array_like [day]
        Array of times in days.
    P : numeric [day]
        Value of the period in days.
    n_terms : int (optional)
        Number of terms in the Fourier series.

    Returns
    -------
    f : `numpy.ndarray` [au/day]
        Sinusoid model computed at each time, `t` in units of au/day
    """
    p = np.array(p)
    A = design_matrix(t, P)
    return (p[np.newaxis] * A).sum(axis=-1)

def period_grid(data, P_min=1, P_max=1E4, resolution=2):
    """

    Parameters
    ----------
    data : `ebak.singleline.RVData`
        Instance of `RVData` containing the data to fit.
    P_min : numeric (optional)
        Minimum period value for the grid.
    P_max : numeric (optional)
        Maximum period value for the grid.
    resolution : numeric (optional)
        Extra factor used in computing the grid spacing.

    Returns
    -------
    P_grid : `numpy.ndarray` [day]
        Grid of periods in days.
    dP_grid : `numpy.ndarray` [day]
        Grid of period spacings in days.
    """
    T_max = data._t.max() - data._t.min()

    def _grid_element(P):
        return P**2 / (2*np.pi*T_max) / resolution

    P_grid = [P_min]
    while np.max(P_grid) < P_max:
        dP = _grid_element(P_grid[-1])
        P_grid.append(P_grid[-1] + dP)

    P_grid = np.array(P_grid)
    dP_grid = _grid_element(P_grid)

    return P_grid, dP_grid

def tensor_vector_scalar(P, data):
    A = design_matrix(data._t, P)
    ATCinv = (A.T * data._ivar[None])
    ATA = ATCinv.dot(A)

    # Note: this is unstable!
    p = np.linalg.solve(ATA, ATCinv.dot(data._rv))
    # if cond num is high, do:
    # p,*_ = np.linalg.lstsq(A, y)

    dy = sinusoid_model(p, data._t, P) - data._rv
    chi2 = np.sum(dy**2 * data._ivar)

    return ATA, p, chi2

def marginal_ln_likelihood(P, data):
    ATA,p,chi2 = tensor_vector_scalar(P, data)

    sign,logdet = np.linalg.slogdet(ATA)
    assert sign == 1.

    return -0.5*chi2 + 0.5*logdet

def sample_posterior(P_grid, probs, size=1):
    norm_cumsum = np.cumsum(probs) / probs.sum()
    inv_cdf = interp1d(norm_cumsum, P_grid, kind='linear')
    return inv_cdf(np.random.uniform(size=size))
