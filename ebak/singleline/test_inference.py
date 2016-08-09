from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.time as atime
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..celestialmechanics_class import SimulatedRVOrbit
from .data import RVData
from .inference import OrbitModel

def test_orbitmodel():
    # Functional test of OrbitModel
    # - TODO: write unit tests

    # simulate an RV curve given an orbit
    true_orbit = SimulatedRVOrbit(P=521.*u.day, a_sin_i=2*u.au, ecc=0.3523,
                                  omega=21.85*u.degree, phi0=-11.723*u.degree,
                                  v0=27.41*u.km/u.s)
    assert (true_orbit.m_f > 1.*u.Msun) and (true_orbit.m_f < 10.*u.Msun) # sanity check

    # generate RV curve
    t = np.random.uniform(55555., 57012., size=32)
    rv = true_orbit.generate_rv_curve(t)

    err = np.full(t.shape, 0.3) * u.km/u.s
    rv += np.random.normal(size=t.shape) * err

    data = RVData(t=t, rv=rv, ivar=1/err**2)

    # data.plot()
    # plt.show()

    # create an orbit slightly off from true
    orbit = true_orbit.copy()
    orbit._P *= 1.1
    orbit.ecc *= 0.9
    orbit._v0 *= 1.1

    assert (OrbitModel(data=data, orbit=true_orbit).ln_posterior() >
            OrbitModel(data=data, orbit=orbit).ln_posterior())

    model = OrbitModel(data=data, orbit=orbit)
    print(model.get_par_vec())

