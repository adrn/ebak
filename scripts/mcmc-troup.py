"""
Run MCMC on RV curves for all Troup stars.

TODO:
- Write out chain to an HDF5 file
- Preserve initial orbit guess
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
import matplotlib.pyplot as plt
import numpy as np
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
    ecc = row['ECC'][0]
    m_f = row['MASSFN'][0]*u.Msun
    K = row['SEMIAMP'][0]*u.m/u.s

    period = row['PERIOD'][0]*u.day
    asini = (K * period/(2*np.pi) * np.sqrt(1 - ecc**2)).to(u.au)

    omega = row['OMEGA'][0]*u.degree
    v0 = row['V0']*u.m/u.s
    v_slope = row['SLOPE'][0]*u.m/u.s/u.day

    t_peri = atime.Time(row['TPERI'][0], format='jd', scale='tcb')
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

    if not res.success:
        raise ValueError("Failed to optimize orbit parameters.")

    orbit = troup_orbit.copy()
    orbit._a_sin_i, orbit._omega, orbit._phi0, orbit._v0 = res.x

    return orbit

def main(apogee_id, n_walkers, n_steps):
    # load data files -- Troup catalog and full APOGEE allVisit file
    _troup = np.genfromtxt(TROUP_DATA_PATH, delimiter=",",
                           names=True, dtype=None)
    troup = tbl.Table(_troup[_troup['APOGEE_ID'].astype(str) == apogee_id])

    _allvisit = fits.getdata(ALLVISIT_DATA_PATH, 1)
    target = tbl.Table(_allvisit[_allvisit['APOGEE_ID'].astype(str) == apogee_id])

    # read data and orbit parameters and produce initial guess for MCMC
    logger.debug("Reading data from Troup catalog and allVisit files...")
    data = allVisit_to_rvdata(target)
    troup_orbit = troup_to_orbit(troup, data)

    # first figure is initial guess
    logger.debug("Plotting initial guess...")
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    data.plot(ax=ax)
    troup_orbit.plot(ax=ax)

    _tdiff = data._t.max() - data._t.min()
    t = np.linspace(data._t.min() - _tdiff*0.1,
                    data._t.max() + _tdiff*0.1, 1024)
    ax.set_xlim(t.min(), t.max())

    ax.set_title("Initial guess orbit")
    ax.set_xlabel("time [BJD]")
    ax.set_ylabel("RV [km/s]")
    fig.tight_layout()
    fig.savefig(join(PLOT_PATH, "{}-0-initial.png".format(apogee_id)), dpi=256)

    # create model object to evaluate prior, likelihood, posterior
    model = OrbitModel(data=data, orbit=troup_orbit.copy())

    # sample initial conditions for walkers
    logger.debug("Sampling initial conditions for MCMC walkers...")
    p0 = emcee.utils.sample_ball(model.get_par_vec(),
                                 1E-3*model.get_par_vec(),
                                 size=n_walkers)

    # special treatment for ln_P
    p0[:,0] = np.random.normal(np.log(model.orbit._P), 0.5, size=p0.shape[0])

    # special treatment for s
    p0[:,6] = np.abs(np.random.normal(0, 1E-3, size=p0.shape[0]) * u.km/u.s).decompose(usys).value

    sampler = emcee.EnsembleSampler(n_walkers, dim=p0.shape[1], lnpostfn=model)

    # make sure all initial conditions return finite probabilities
    for pp in p0:
        assert np.isfinite(model(pp))

    # run the damn sampler!
    logger.debug("Running the MCMC sampler for {} steps...".format(n_steps))
    pos,_,_ = sampler.run_mcmc(p0, N=n_steps)

    # plot orbits computed from the samples
    logger.debug("Plotting the MCMC samples...")
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    fig = model.plot_rv_samples(sampler, ax=ax)
    _ = model.data.plot(ax=ax)
    fig.tight_layout()
    fig.savefig(join(PLOT_PATH, "{}-1-rv-curves.png".format(apogee_id)), dpi=256)

    # make a corner plot
    flatchain = np.vstack(sampler.chain[:,-256:])
    plot_pars = model.vec_to_plot_pars(flatchain)
    troup_vals = [np.log(troup_orbit.P.to(u.day).value), troup_orbit.m_f.value,
                  troup_orbit.ecc, troup_orbit.omega.to(u.degree).value,
                  troup_orbit.t0.mjd, -troup_orbit.v0.to(u.km/u.s).value, 0.]
    fig = corner.corner(plot_pars, labels=model.plot_labels, truths=troup_vals)
    fig.savefig(join(PLOT_PATH, "{}-2-corner.png".format(apogee_id)), dpi=256)
    logger.debug("done!")

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

    parser.add_argument("--n-steps", dest="n_steps", default=4096,
                        type=int, help="Number of MCMC steps")
    parser.add_argument("--n-walkers", dest="n_walkers", default=256,
                        type=int, help="Number of MCMC walkers")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.apogee_id, n_walkers=args.n_walkers, n_steps=args.n_steps)
