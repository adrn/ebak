from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
from astropy.constants import c
import numpy as np

c_kms = c.to(u.km/u.s).value

def get_design_matrix(data_spec, ref_spec, v1, v2):
    """
    Input are two spectra as dictionaries: the data spectrum must have
    'wvln' and 'flux' keys and the reference spectrum must have an 'interp'
    key that contains a interpolating function that evaluates the reference
    spectrum on a given wavelength grid.

    Note: Positive velocity is a redshift.

    Parameters
    ----------
    data_spec : dict
    ref_spec : dict
    v1 : float
        In km/s
    v2 : float
        In km/s

    Returns
    -------
    X : numpy.ndarray
        Design matrix.
    """
    X = np.ones((3, data_spec['wvln'].shape[0]))
    X[1] = ref_spec['interp'](data_spec['wvln'] * (1 + v1/c_kms)) # this is only good to first order in (v/c)
    X[2] = ref_spec['interp'](data_spec['wvln'] * (1 + v2/c_kms))

    # HACK: where there are extrpolation errors, set to zero and make error infinite
    any_nan = np.isnan(X).any(axis=0)
    if any_nan.any():
        X[:,any_nan] = 0.
        ref_spec['flux_err'] = np.inf

    return X

def get_optimal_chisq_pars(data_spec, ref_spec, v1, v2):
    X = get_design_matrix(data_spec, ref_spec, v1, v2)
    C_inv_diag = 1 / (data_spec['flux_err']**2 + ref_spec['flux_err']**2) + 1e-15 # HACK: magic number to prevent singular matrix
    return np.linalg.solve((X * C_inv_diag[None]).dot(X.T),
                           (X * C_inv_diag[None]).dot(data_spec['flux']))

def make_synthetic_spectrum(X, pars):
    """
    Make a synthesized (model) spectrum from a design matrix and the
    nuisance parameters. For example, the parameters might be a shift
    and two scales.
    """
    return X.T.dot(pars)

def ln_likelihood(v1_v2, data_spec, ref_spec):
    """
    Input are two spectra as dictionaries: the data spectrum must have
    'wvln' and 'flux' keys and the reference spectrum must have an 'interp'
    key that contains a interpolating function that evaluates the reference
    spectrum on a given wavelength grid.

    Parameters
    ----------
    v1_v2 : iterable
        The two velocities, v1, v2
    data_spec : dict
    ref_spec : dict

    Returns
    -------
    lnlikelihood : float

    """
    v1,v2 = v1_v2
    X = get_design_matrix(data_spec, ref_spec, v1, v2)
    inv_var = 1 / (data_spec['flux_err']**2 + ref_spec['flux_err']**2)
    C_inv_diag = inv_var + 1e-15 # HACK: magic number to prevent singular matrix

    X_Cinv_XT = (X * C_inv_diag[None]).dot(X.T)
    optimal_pars = np.linalg.solve(X_Cinv_XT,
                                   (X * C_inv_diag[None]).dot(data_spec['flux']))

    model_spec = make_synthetic_spectrum(X, optimal_pars)
    _,logdet = np.linalg.slogdet(X_Cinv_XT)
    return -0.5 * np.sum((data_spec['flux'] - model_spec)**2 * inv_var) - 0.5*logdet
