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

class RVOrbit(object):
    """

    Parameters
    ----------
    P : `~astropy.units.Quantity` [time]
        Orbital period.
    a : `~astropy.units.Quantity` [length]
        Semi-major axis.
    sin_i : numeric
        Sin of the inclination angle.
    ecc : numeric
        Eccentricity.
    omega : `~astropy.units.Quantity` [angle]
        Argument of perihelion.
    t0 : `~astropy.time.Time`
        Time of pericenter.
    v0 : `~astropy.units.Quantity` [speed]
        Systemic velocity
    """
    @u.quantity_input(P=u.yr, a=u.au, omega=u.radian, v0=u.km/u.s)
    def __init__(self, P, a, sin_i, ecc, omega, t0, v0=0*u.km/u.s):
        # store unitful things without units for speed
        self._P = P.decompose(usys).value
        self._a = a.decompose(usys).value
        self._omega = omega.decompose(usys).value
        if isinstance(t0, at.Time):
            self._t0 = t0.tcb.mjd
        else:
            self._t0 = t0
        self._v0 = v0.decompose(usys).value

        self.sin_i = float(sin_i)
        self.ecc = float(ecc)

    @property
    def P(self):
        return self._P * usys['time']

    @property
    def a(self):
        return self._a * usys['length']

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

    def set_par_from_vec(self, p):
        (self._P, self._a, self.sin_i, self.ecc,
         self._omega, self._t0, self._v0) = p

    def get_par_vec(self):
        return np.array([self._P, self._a, self.sin_i, self.ecc,
                         self._omega, self._t0, self._v0])

class SimulatedRVOrbit(RVOrbit):

    def generate_rv_curve(self, t):
        """
        Parameters
        ----------
        t : array_like, `~astropy.time.Time`
            Array of times. Either in BJD or as an Astropy time.
        """

        if isinstance(t, at.Time):
            _t = t.tcb.mjd
        else:
            _t = t

        rv = rv_from_elements(_t, self._P, self._a, self.sin_i,
                              self.ecc, self._omega, self._t0,
                              self._v0)
        return (rv*self.units['speed']).to(u.km/u.s)

    def plot(self, t=None, ax=None):
        """
        needs t or ax

        TODO: add kwargs support
        """
        if t is None and ax is None:
            raise ValueError("You must pass a time array (t) or axes "
                             "instance (ax)")

        if ax is None:
            fig,ax = plt.subplots(1,1)

        if t is None:
            t = np.linspace(*ax.get_xlim(), 1024)

        rv = self.generate_rv_curve(t)
        ax.plot(t, rv.value,
                linestyle='-', alpha=0.5, marker=None, color='r')

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

    def plot(self, ax=None):
        """
        TODO: add kwargs support
        """
        if ax is None:
            fig,ax = plt.subplots(1,1)

        ax.errorbar(self.t.value, self.rv.to(u.km/u.s).value,
                    self.stddev.to(u.km/u.s).value,
                    linestyle='none', alpha=1., marker='o', color='k',
                    ecolor='#666666')

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
        rvs = self.orbit.generate_rv_curve(self.data._t)
        return -0.5 * self.data._ivar * (self.data._rv - rvs)**2

    def __call__(self, p):
        self.model.set_par_from_vec(p)
        return self.ln_likelihood()
