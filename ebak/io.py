from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.io import fits
import numpy as np

def load_apVisit(filename, chip=None):
    """
    Load an apVisit file into a dictionary structure. If specified, only load data from one chip.

    Parameters
    ----------
    filename : str
        Path to the apVisit file.
    chip : int (optional)
        The index of the chip to load. Can be 0,1,2 going from red to blue
        (0 is reddest, 2 is bluest).
    """
    hdulist1 = fits.open(filename)
    wvln = hdulist1[4].data
    flux = hdulist1[1].data
    flux_err = hdulist1[2].data

    if chip is not None:
        return {'wvln': wvln[chip], 'flux': flux[chip], 'flux_err': flux_err[chip]}
    else:
        return {'wvln': wvln, 'flux': flux, 'flux_err': flux_err}


def load_apStar(filename):
    """
    Load an apStar file into a dictionary structure.

    Parameters
    ----------
    filename : str
        Path to the apStar file.
    """
    hdulist1 = fits.open(filename)
    flux = hdulist1[1].data[0]
    flux_err = hdulist1[2].data[0]
    wvln = 10**(hdulist1[0].header['CRVAL1'] + np.arange(flux.size) * hdulist1[0].header['CDELT1'])

    return {'wvln': wvln, 'flux': flux, 'flux_err': flux_err}
