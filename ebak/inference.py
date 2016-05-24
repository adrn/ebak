from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
from astropy.constants import c
import numpy as np


@u.quantity_input(v1=u.km/u.s, v2=u.km/u.s)
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
    v1 : astropy.units.Quantity
    v2 : astropy.units.Quantity

    Returns
    -------
    X : numpy.ndarray
        Design matrix.
    """
    X = np.ones((3, data_spec['wvln'].shape[0]))
    X[1] = ref_spec['interp'](data_spec['wvln'] * (1 + v1/c)) # this is only good to first order in (v/c)
    X[2] = ref_spec['interp'](data_spec['wvln'] * (1 + v2/c))
    return X

@u.quantity_input(v1=u.km/u.s, v2=u.km/u.s)
def get_optimal_chisq(data_spec, ref_spec, v1, v2):
    X = get_design_matrix(data_spec, ref_spec, v1, v2)
    return np.linalg.solve(X.dot(X.T), X.dot(data_spec['flux']))

def make_synthetic_spectrum(X, pars):
    """
    Make a synthesized (model) spectrum from a design matrix and the
    nuisance parameters. For example, the parameters might be a shift
    and two scales.
    """
    return X.T.dot(pars)

