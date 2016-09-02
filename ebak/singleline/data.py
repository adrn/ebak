from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import astropy.time as at
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import six

# Project
from ..units import usys

__all__ = ['RVData']

class RVData(object):
    """
    Parameters
    ----------
    t : array_like, `~astropy.time.Time`
        Array of times. Either in BJD or as an Astropy time.
    rv : `~astropy.units.Quantity` [speed]
        Radial velocity measurements.
    ivar : `~astropy.units.Quantity` [1/speed^2]
    """
    @u.quantity_input(rv=u.km/u.s, ivar=(u.s/u.km)**2)
    def __init__(self, t, rv, ivar):
        if isinstance(t, at.Time):
            _t = t.tcb.mjd
        else:
            _t = t
        self._t = _t

        self._rv = rv.decompose(usys).value
        self._ivar = ivar.decompose(usys).value

    @property
    def t(self):
        return at.Time(self._t, scale='tcb', format='mjd')

    @property
    def phase(self, t0, P):
        return ((self.t - t0) / P) % 1.

    @property
    def rv(self):
        return self._rv * usys['length'] / usys['time']

    @property
    def ivar(self):
        return self._ivar * (usys['time'] / usys['length'])**2

    @property
    def stddev(self):
        return 1 / np.sqrt(self.ivar)

    # ---

    def plot(self, ax=None, **kwargs):
        """
        """
        if ax is None:
            fig,ax = plt.subplots(1,1)

        style = kwargs.copy()
        style.setdefault('linestyle', 'none')
        style.setdefault('alpha', 1.)
        style.setdefault('marker', 'o')
        style.setdefault('color', 'k')
        style.setdefault('ecolor', '#666666')

        ax.errorbar(self.t.value, self.rv.to(u.km/u.s).value,
                    self.stddev.to(u.km/u.s).value, **style)

        return ax

    # copy methods
    def __copy__(self):
        return self.__class__(t=self.t.copy(),
                              rv=self.rv.copy(),
                              ivar=self.ivar.copy())
    def copy(self):
        return self.__copy__()

    def __getitem__(self, slc):
        return self.__class__(t=self.t.copy()[slc],
                              rv=self.rv.copy()[slc],
                              ivar=self.ivar.copy()[slc])

    def __len__(self):
        return len(self._t)

    @classmethod
    def from_apogee(cls, path_or_data, apogee_id=None):
        """
        Parameters
        ----------
        apogee_id : str
            The APOGEE ID of the desired target, e.g., 2M03080601+7950502.
        path_or_data : str, numpy.ndarray
            Either a string path to the location of the APOGEE allVisit
            FITS file, or a selection of rows from the allVisit file.
        """

        if isinstance(path_or_data, six.string_types):
            _allvisit = fits.getdata(path_or_data, 1)
            target = _allvisit[_allvisit['APOGEE_ID'].astype(str) == apogee_id]
        else:
            target = path_or_data

        rv = np.array(target['VHELIO']) * u.km/u.s
        ivar = 1 / (np.array(target['VRELERR'])**2 * (u.km/u.s)**2)
        t = at.Time(np.array(target['JD']), format='jd', scale='tcb')

        idx = np.isfinite(rv.value) & np.isfinite(t.value) & np.isfinite(ivar.value)
        return cls(t[idx], rv[idx], ivar[idx])
