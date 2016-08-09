from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.time as atime
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from .data import RVData
from .inference import OrbitModel

def test_orbitmodel():

    # test various initializations
    t = np.random.uniform(55555., 56012., size=1024)
    rv = (21 * np.sin(2*np.pi*t/51.412) - 49.) * u.km/u.s
    ivar = 1 / (np.random.normal(0,5,size=1024)*u.km/u.s)**2
    data = RVData(t=t, rv=rv, ivar=ivar)

    model = OrbitModel(data=data)
    model()

    # TODO: write unit tests

