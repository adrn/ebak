""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import astropy.time as at
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from gala.units import UnitSystem

# Project
from .celestialmechanics import rv_from_elements

__all__ = ['RVOrbit', 'RVData', 'SimulatedRVOrbit', 'OrbitModel']

usys = UnitSystem(u.au, u.day, u.radian, u.Msun)
_ivar_disk = (1 / (30.*u.km/u.s)**2).decompose(usys).value # used in ln_prior below

class RVOrbit(object):
    """

    Parameters
    ----------
    P : `~astropy.units.Quantity` [time]
        Orbital period.
    a_sin_i : `~astropy.units.Quantity` [length]
        Semi-major axis times sine of inclination angle.
    ecc : numeric
        Eccentricity.
    omega : `~astropy.units.Quantity` [angle]
        Argument of perihelion.
    t0 : `~astropy.time.Time`
        Time of pericenter.
    v0 : `~astropy.units.Quantity` [speed]
        Systemic velocity
    """
    @u.quantity_input(P=u.yr, a_sin_i=u.au, omega=u.radian, v0=u.km/u.s)
    def __init__(self, P, a_sin_i, ecc, omega, t0, v0=0*u.km/u.s):
        # store unitful things without units for speed
        self._P = P.decompose(usys).value
        self._a_sin_i = a.decompose(usys).value
        self._omega = omega.decompose(usys).value
        if isinstance(t0, at.Time):
            self._t0 = t0.tcb.mjd
        else:
            self._t0 = t0
        self._v0 = v0.decompose(usys).value

        self.ecc = float(ecc)

    @property
    def P(self):
        return self._P * usys['time']

    @property
    def a_sin_i(self):
        return self._a_sin_i * usys['length']

    @property
    def omega(self):
        return self._omega * usys['angle']

    @property
    def t0(self):
        return at.Time(self._t0, scale='tcb', format='mjd')

    @property
    def v0(self):
        return self._v0 * usys['length'] / usys['time']

    @property
    def units(self):
        return usys

    @classmethod
    def from_vec(cls, p):
        (_P, _a_sin_i, sqrte_cos_pomega,
         sqrte_sin_pomega, _t0, _v0) = p
        ecc = sqrte_cos_pomega**2 + sqrte_sin_pomega**2
        _omega = np.arctan2(sqrte_sin_pomega, sqrte_cos_pomega)
        return cls(P=_P*usys['time'], a_sin_i=_a_sin_i*usys['length'],
                   ecc=ecc, omega=_omega*usys['angle'], t0=_t0,
                   v0=_v0*usys['length']/usys['time'])

    def set_par_from_vec(self, p):
        (ln_P, self._a_sin_i, sqrte_cos_pomega,
         sqrte_sin_pomega, self._t0, self._v0) = p
        self.ecc = sqrte_cos_pomega**2 + sqrte_sin_pomega**2
        self._omega = np.arctan2(sqrte_sin_pomega, sqrte_cos_pomega)
        self._P = np.exp(ln_P)

    def get_par_vec(self):
        return np.array([np.log(self._P), self._a_sin_i,
                         np.sqrt(self.ecc)*np.cos(self._omega),
                         np.sqrt(self.ecc)*np.sin(self._omega),
                         self._t0, self._v0])

class SimulatedRVOrbit(RVOrbit):

    def _generate_rv_curve(self, t):
        """
        Parameters
        ----------
        t : array_like, `~astropy.time.Time`
            Array of times. Either in BJD or as an Astropy time.

        Returns
        -------
        rv : numpy.ndarray
        """

        if isinstance(t, at.Time):
            _t = t.tcb.mjd
        else:
            _t = t

        rv = rv_from_elements(_t, self._P, self._a_sin_i,
                              self.ecc, self._omega, self._t0,
                              self._v0)
        return rv

    def generate_rv_curve(self, t):
        """
        Parameters
        ----------
        t : array_like, `~astropy.time.Time`
            Array of times. Either in BJD or as an Astropy time.

        Returns
        -------
        rv : astropy.units.Quantity [km/s]
        """
        rv = self._generate_rv_curve(t)
        return (rv*self.units['speed']).to(u.km/u.s)

    def plot(self, t=None, ax=None, **kwargs):
        """
        needs t or ax
        """
        if t is None and ax is None:
            raise ValueError("You must pass a time array (t) or axes "
                             "instance (ax)")

        if ax is None:
            fig,ax = plt.subplots(1,1)

        if t is None:
            t = np.linspace(*ax.get_xlim(), 1024)

        style = kwargs.copy()
        style.setdefault('linestyle', '-')
        style.setdefault('alpha', 0.5)
        style.setdefault('marker', None)
        style.setdefault('color', 'r')

        rv = self.generate_rv_curve(t)
        ax.plot(t, rv.value, **style)

        return ax


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
        TODO: add kwargs support
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


class OrbitModel(object):
    """
    Parameters
    ----------
    orbit : RVOrbit
    data : RVData

    """
    def __init__(self, orbit, data):
        self.orbit = orbit
        self.data = data

    def ln_likelihood(self):
        rvs = self.orbit._generate_rv_curve(self.data._t)
        return -0.5 * self.data._ivar * (self.data._rv - rvs)**2

    def ln_prior(self):
        lnp = 0.

        # assumes sampler is stepping in log(P)
        if self.orbit._P < 1. or self.orbit._P > 8192*365.: # days
            return -np.inf

        if 1E-6 < self.orbit._a_sin_i < 16384.: # au
            lnp += -np.log(self.orbit._a_sin_i)
        else:
            return -np.inf

        if self.orbit.ecc < 0. or self.orbit.ecc > 1.:
            return -np.inf

        # HACK: we could remove the cyclic variable t0
        _t0_epoch = 55000.
        if (self.orbit._t0 < (_t0_epoch-1.5*self.orbit._P) or
            self.orbit._t0 < (_t0_epoch-1.5*self.orbit._P)):
            return -np.inf

        # Gaussian with velocity dispersion of the disk
        lnp += -0.5 * _ivar_disk * self.orbit._v0**2

        return lnp

    def ln_posterior(self):
        lnp = self.ln_prior()
        if not np.isfinite(lnp):
            return -np.inf
        return lnp + self.ln_likelihood().sum()

    def __call__(self, p):
        self.orbit.set_par_from_vec(p)
        return self.ln_posterior()