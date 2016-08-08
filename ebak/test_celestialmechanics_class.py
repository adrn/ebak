from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.time as at
import astropy.units as u
import numpy as np
import pytest

from .celestialmechanics_class import SimulatedRVOrbit

def test_simulatedrvorbit():
    t0 = at.Time(55555., scale='utc', format='mjd')
    orbit = SimulatedRVOrbit(P=30.*u.day, a=0.1*u.au, sin_i=0.4,
                             ecc=0.1, omega=0.*u.radian, t0=t0)

    t = np.random.uniform(55612., 55792, 128)
    rv = orbit.generate_rv_curve(t)
