""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import astropy.time as at
import astropy.units as u
import numpy as np
from gala.units import UnitSystem

# Project
from .celestialmechanics import rv_from_elements

usys = UnitSystem(u.au, u.day, u.radian, u.Msun)

class OrbitModel(object):
    """

    Parameters
    ----------
    P : `~astropy.units.Quantity`
        Orbital period.
    a : `~astropy.units.Quantity`
        Semi-major axis.
    sin_i : numeric
        Sin of the inclination angle.
    ecc : numeric
        Eccentricity.
    omega : `~astropy.units.Quantity`
        Argument of perihelion.
    t0 : `~astropy.time.Time`
        Time of pericenter.
    v0 : `~astropy.units.Quantity`
        Systemic velocity
    """
    @u.quantity_input(P=u.yr, a=u.au, omega=u.radian, v0=u.km/u.s)
    def __init__(self, P, a, sin_i, ecc, omega, t0, v0=0*u.km/u.s):
        # store unitful things without units for speed
        self._P = P.decompose(usys).value
        self._a = a.decompose(usys).value
        self._omega = omega.decompose(usys).value
        self._t0 = t0.tcb.mjd
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

    @classmethod
    def par_from_arr(self, p):
        pass

    @classmethod
    def arr_from_par(self, P, a, sin_i, ecc, omega, t0, systemic_velocity):
        pass


class SimulatedRVOrbit(OrbitModel):

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
                              self.ecc, self._omega, self._t0)
        return (rv*self.units['speed']).to(u.km/u.s)

