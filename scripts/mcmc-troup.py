"""
Run MCMC on all Troup stars...
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
from os.path import abspath, join, split, exists

# Third-party
from astropy import log as logger
from astropy.io import fits, ascii
import astropy.table as tbl
import astropy.time as atime
import astropy.coordinates as coord
import astropy.units as u

import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('apw-notebook')
import corner
from scipy.optimize import minimize

# Project
from ebak import SimulatedRVOrbit
from ebak.singleline import RVData, OrbitModel
from ebak.units import usys

_basepath = split(abspath(join(__file__, "..")))[0]

# Hacks: Hard-set paths
TROUP_DATA_PATH = join(_basepath, "data", "troup16-dr12.csv")
ALLVISIT_DATA_PATH = join(_basepath, "data", "allVisit-l30e.2.fits")
PLOT_PATH = join(_basepath, "plots", "troup")
if not exists(PLOT_PATH):
    os.mkdir(PLOT_PATH)

def allVisit_to_rvdata(rows):
    rv = np.array(rows['VHELIO']) * u.km/u.s
    ivar = 1 / (np.array(rows['VRELERR'])*u.km/u.s)**2
    t = atime.Time(np.array(rows['JD']), format='jd', scale='tcb')
    return RVData(t, rv, ivar)

def troup_to_orbit(row, data):
    ecc = row['ECC']
    m_f = row['MASSFN']*u.Msun
    K = row['SEMIAMP']*u.m/u.s

    period = row['PERIOD']*u.day
    asini = (K * period/(2*np.pi) * np.sqrt(1 - ecc**2)).to(u.au)

    omega = row['OMEGA']*u.degree
    v0 = row['V0']*u.m/u.s
    v_slope = row['SLOPE']*u.m/u.s/u.day

    t_peri = atime.Time(row['TPERI'], format='jd', scale='tcb')
    phi0 = ((2*np.pi*(t_peri.tcb.mjd - 55555.) / period.to(u.day).value) % (2*np.pi)) * u.radian

    # now, because we don't believe anything, we'll take the period and
    #   eccentricity as fixed and optimize to get all other parameters
    troup_orbit = SimulatedRVOrbit(P=period, a_sin_i=asini, ecc=ecc,
                                   omega=omega, phi0=phi0, v0=0*u.km/u.s)

    def min_func(p, data, _orbit):
        a_sin_i, omega, phi0, v0 = p

        _orbit._a_sin_i = a_sin_i
        _orbit._omega = omega
        _orbit._phi0 = phi0
        _orbit._v0 = v0

        return np.sum(data._ivar * (_orbit._generate_rv_curve(data._t) - data._rv)**2)

    x0 = [asini.decompose(usys).value, omega.decompose(usys).value,
          phi0.decompose(usys).value, -v0.decompose(usys).value]
    res = minimize(min_func, x0=x0, method='powell',
                   args=(data,troup_orbit.copy()))

    if res.success is False:
        raise ValueError("Failed to optimize orbit parameters.")

    orbit = troup_orbit.copy()
    orbit._a_sin_i, orbit._omega, orbit._phi0, orbit._v0 = res.x

    return orbit

def main(apogee_id):
    # load data files -- Troup catalog and full APOGEE allVisit file
    _troup = np.genfromtxt(TROUP_DATA_PATH, delimiter=",",
                           names=True, dtype=None)
    troup = tbl.Table(_troup[_troup['APOGEE_ID'].astype(str) == apogee_id])

    _allvisit = fits.getdata(ALLVISIT_DATA_PATH, 1)
    target = tbl.Table(_allvisit[_allvisit['APOGEE_ID'].astype(str) == apogee_id])

    # read data and orbit parameters and produce initial guess for MCMC
    data = allVisit_to_rvdata(target)
    orbit = troup_to_orbit(troup, data)

    # first figure is initial guess
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    data.plot(ax=ax)
    orbit.plot(ax=ax)
    ax.set_title("Initial guess orbit")
    ax.set_xlabel("time [BJD]")
    ax.set_ylabel("RV [km/s]")
    fig.tight_layout()
    fig.savefig(join(PLOT_PATH, "{}-0-initial.png".format(apogee_id)))

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("--id", dest="apogee_id", default=None, required=True,
                        type=str, help="APOGEE ID")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.apogee_id)
