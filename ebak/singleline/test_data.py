from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.time as atime
import astropy.units as u
import numpy as np
import pytest

from .data import RVData

def test_rvdata():

    # test various initializations
    t = np.random.uniform(55555., 56012., size=1024)
    rv = 100 * np.sin(0.5*t) * u.km/u.s
    ivar = 1 / (np.random.normal(0,5,size=1024)*u.km/u.s)**2
    RVData(t=t, rv=rv, ivar=ivar)

    t = atime.Time(t, format='mjd', scale='utc')
    RVData(t=t, rv=rv, ivar=ivar)

    with pytest.raises(TypeError):
        RVData(t=t, rv=rv.value, ivar=ivar)

    with pytest.raises(TypeError):
        RVData(t=t, rv=rv, ivar=ivar.value)
