"""
Run MCMC on RV curves for all Troup stars.

TODO:
- Write out chain to an HDF5 file
- Preserve initial orbit guess
- If eccentric_anomaly_from_mean_anomaly() raises a warning, return -np.inf?
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
from os.path import abspath, join, split, exists
import time
import sys

# Third-party
from astropy import log as logger
from astropy.io import fits, ascii
import astropy.table as tbl
import astropy.time as atime
import astropy.coordinates as coord
import astropy.units as u
import h5py

import emcee
import kombine
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('apw-notebook')
import corner
from scipy.optimize import minimize
from gala.util import get_pool

# Project
from ebak import SimulatedRVOrbit, EPOCH
from ebak.singleline import RVData, OrbitModel
from ebak.units import usys

_basepath = split(abspath(join(__file__, "..")))[0]

# Hacks: Hard-set paths
TROUP_DATA_PATH = join(_basepath, "data", "troup16-dr12.csv")
ALLVISIT_DATA_PATH = join(_basepath, "data", "allVisit-l30e.2.fits")
PLOT_PATH = join(_basepath, "plots", "troup")
OUTPUT_PATH = join(_basepath, "output")
for PATH in [PLOT_PATH, OUTPUT_PATH]:
    if not exists(PATH):
        os.makedirs(PATH)

def allVisit_to_rvdata(rows):
    rv = np.array(rows['VHELIO']) * u.km/u.s
    ivar = 1 / (np.array(rows['VRELERR'])*u.km/u.s)**2
    t = atime.Time(np.array(rows['JD']), format='jd', scale='tcb')
    idx = np.isfinite(rv.value) & np.isfinite(t.value) & np.isfinite(ivar.value)
    return RVData(t[idx], rv[idx], ivar[idx])

def troup_to_init_orbit(row, data):
    ecc = row['ECC'][0]
    mf = row['MASSFN'][0]*u.Msun
    K = row['SEMIAMP'][0]*u.m/u.s

    period = row['PERIOD'][0]*u.day
    asini = (K * period/(2*np.pi) * np.sqrt(1 - ecc**2)).to(u.au)

    omega = row['OMEGA'][0]*u.degree
    v0 = row['V0']*u.m/u.s
    v_slope = row['SLOPE'][0]*u.m/u.s/u.day

    t0 = atime.Time(row['T0'][0], format='jd', scale='tcb')
    phi0 = 2*np.pi * (((t0.tcb.mjd - EPOCH) / period.to(u.day).value) % 1.) * u.radian

    # now, because we don't believe anything, we'll take the period and
    #   eccentricity as fixed and optimize to get all other parameters
    troup_orbit = SimulatedRVOrbit(P=period, a_sin_i=asini, ecc=ecc,
                                   omega=omega, phi0=phi0, v0=0*u.km/u.s)

    def min_func(p, data, _orbit):
        a_sin_i, omega, phi0, v0 = p
        # omega, phi0, v0 = p

        _orbit._a_sin_i = a_sin_i
        _orbit._omega = omega
        _orbit._phi0 = phi0
        _orbit._v0 = v0

        return np.sum(data._ivar * (_orbit._generate_rv_curve(data._t) - data._rv)**2)

    # x0 = [omega.decompose(usys).value,
    #       phi0.decompose(usys).value,
    #       np.median(data._rv)]
    x0 = [asini.decompose(usys).value,
          omega.decompose(usys).value,
          phi0.decompose(usys).value,
          -np.median(data._rv)]
    res = minimize(min_func, x0=x0, method='powell',
                   args=(data,troup_orbit.copy()))

    if not res.success:
        raise ValueError("Failed to optimize orbit parameters.")

    orbit = troup_orbit.copy()
    orbit._a_sin_i, orbit._omega, orbit._phi0, orbit._v0 = res.x
    # orbit._omega, orbit._phi0, orbit._v0 = res.x

    return orbit

def plot_init_orbit(orbit, data, apogee_id):
    logger.debug("Plotting initial guess...")

    # TODO: make these arguments?
    data_style = dict(marker='o', ecolor='#666666', linestyle='none',
                      alpha=0.75, color='k', label='APOGEE data')
    model_style = dict(marker=None, linestyle='-', color='#de2d26',
                       alpha=0.6, label='Troup orbit')

    fig,ax = plt.subplots(1,1,figsize=(8,6))

    # data points
    data_phase = ((data.t - orbit.t0) / orbit.P) % 1.
    ax.errorbar(data_phase, data.rv.to(u.km/u.s).value,
                data.stddev.to(u.km/u.s).value, **data_style)

    # model curve
    model_t = data.t.min() + \
        atime.TimeDelta(np.linspace(0., orbit.P.value, 1024)*orbit.P.unit)
    model_rv = orbit.generate_rv_curve(model_t).to(u.km/u.s)
    model_phase = ((model_t - orbit.t0) / orbit.P) % 1.
    idx = model_phase.argsort()
    ax.plot(model_phase[idx], model_rv[idx], **model_style)

    ax.set_xlim(-0.1, 1.1)
    ax.set_title("Optimized orbit")
    ax.set_xlabel("phase")
    ax.set_ylabel("RV [km/s]")
    fig.tight_layout()
    fig.savefig(join(PLOT_PATH, "{}-0-initial.png".format(apogee_id)), dpi=256)

def plot_mcmc_diagnostics(sampler, p0, model, sampler_name, apogee_id):

    # Acceptance
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    ax.plot(sampler.acceptance_fraction, 'k', alpha=.5)

    if sampler_name == "kombine":
        ax.set_xlabel("step")
        ax.set_ylabel("mean acceptance fraction")
    elif sampler_name == "emcee":
        ax.set_xlabel("walker num")
        ax.set_ylabel("acceptance fraction")

    fig.tight_layout()
    fig.savefig(join(PLOT_PATH, "{}-3-accep.png".format(apogee_id)), dpi=256)

    # MCMC walker trace
    fig,axes = plt.subplots(p0.shape[1], 1, figsize=(6,3*p0.shape[1]), sharex=True)
    for i in range(p0.shape[1]):
        axes[i].set_ylabel(model.vec_labels[i])
        axes[i].plot(sampler.chain[...,i].T, drawstyle='steps',
                     alpha=0.1, marker=None)
        # axes[i].plot(sampler.chain[:,-1,i], marker='.', alpha=0.25, color='k')

    axes[i].set_xlim(0, p0.shape[0] + 2)
    fig.tight_layout()
    fig.savefig(join(PLOT_PATH, "{}-4-walkers.png".format(apogee_id)), dpi=256)

# ---

def main(apogee_id, index, n_walkers, n_steps, sampler_name, n_burnin=128,
         mpi=False, seed=42, overwrite=False):

    # MPI shite
    pool = get_pool(mpi=mpi, loadbalance=True)
    # need load-balancing - see: https://groups.google.com/forum/#!msg/mpi4py/OJG5eZ2f-Pg/EnhN06Ozg2oJ

    # read in Troup catalog
    _troup = np.genfromtxt(TROUP_DATA_PATH, delimiter=",",
                           names=True, dtype=None)

    if index is not None and apogee_id is None:
        apogee_id = _troup['APOGEE_ID'].astype(str)[index]

    OUTPUT_FILENAME = join(OUTPUT_PATH, "troup-{}.hdf5".format(sampler_name))
    if exists(OUTPUT_FILENAME) and not overwrite:
        with h5py.File(OUTPUT_FILENAME) as f:
            if apogee_id in f.groups():
                logger.info("{} has already been modeled - use '--overwrite' "
                            "to re-run MCMC for this target.")

    # load data files -- Troup catalog and full APOGEE allVisit file
    troup = tbl.Table(_troup[_troup['APOGEE_ID'].astype(str) == apogee_id])
    _allvisit = fits.getdata(ALLVISIT_DATA_PATH, 1)
    target = tbl.Table(_allvisit[_allvisit['APOGEE_ID'].astype(str) == apogee_id])

    # read data and orbit parameters and produce initial guess for MCMC
    logger.debug("Reading data from Troup catalog and allVisit files...")
    data = allVisit_to_rvdata(target)
    troup_orbit = troup_to_init_orbit(troup, data)
    n_dim = 7 # HACK: magic number

    # first figure is initial guess
    plot_init_orbit(troup_orbit, data, apogee_id)

    # create model object to evaluate prior, likelihood, posterior
    model = OrbitModel(data=data, orbit=troup_orbit.copy())

    # sample initial conditions for walkers
    logger.debug("Generating initial conditions for MCMC walkers...")
    p0 = emcee.utils.sample_ball(model.get_par_vec(),
                                 1E-3*model.get_par_vec(),
                                 size=n_walkers)

    # special treatment for ln_P
    p0[:,0] = np.random.normal(np.log(model.orbit._P), 0.5, size=p0.shape[0])

    # special treatment for s
    p0[:,6] = np.abs(np.random.normal(0, 1E-3, size=p0.shape[0]) * u.km/u.s).decompose(usys).value

    if sampler_name == 'emcee':
        sampler = emcee.EnsembleSampler(n_walkers, dim=n_dim,
                                        lnpostfn=model, pool=pool)

    elif sampler_name == 'kombine':
        # TODO: add option for Prior-sampeld initial conditions, don't assume uniform for kombine
        p0 = np.zeros((n_walkers, n_dim))

        p0[:,0] = np.random.uniform(1., 8., n_walkers)

        _asini = np.random.uniform(-1., 3., n_walkers)
        _phi0 = np.random.uniform(0, 2*np.pi, n_walkers)
        p0[:,1] = _asini * np.cos(_phi0)
        p0[:,2] = _asini * np.sin(_phi0)

        _ecc = np.random.uniform(0, 1, n_walkers)
        _omega = np.random.uniform(0, 2*np.pi, n_walkers)
        p0[:,3] = np.sqrt(_ecc) * np.cos(_omega)
        p0[:,4] = np.sqrt(_ecc) * np.sin(_omega)

        p0[:,5] = (np.random.normal(0., 75., n_walkers) * u.km/u.s).decompose(usys).value

        p0[:,6] = (np.exp(np.random.uniform(-8, 0., n_walkers)) * u.km/u.s).decompose(usys).value

        sampler = kombine.Sampler(n_walkers, ndim=n_dim,
                                  lnpostfn=model, pool=pool)

    else:
        raise ValueError("Invalid sampler name '{}'".format(sampler_name))

    # make sure all initial conditions return finite probabilities
    for pp in p0:
        assert np.isfinite(model(pp))

    # burn-in phase
    if n_burnin > 0:
        logger.debug("Burning in the MCMC sampler for {} steps...".format(n_steps))

        if sampler_name == 'kombine':
            sampler.burnin(p0)
        else:
            pos,_,_ = sampler.run_mcmc(p0, N=n_steps)
            sampler.reset()

    else:
        pos = p0

    # run the damn sampler!
    _t1 = time.time()

    logger.info("Running MCMC sampler for {} steps...".format(n_steps))
    if sampler_name == 'kombine':
        sampler.run_mcmc(n_steps)
    else:
        sampler.run_mcmc(pos, N=n_steps)

    pool.close()
    logger.info("done sampling after {} seconds.".format(time.time()-_t1))

    if sampler_name == 'kombine':
        # HACK: kombine uses different axes order
        chain = np.swapaxes(sampler.chain, 0, 1)
    else:
        chain = sampler.chain

    # output the chain and metadata to HDF5 file
    with h5py.File(OUTPUT_FILENAME, 'a') as f: # read/write if exists, create otherwise
        if apogee_id in f and overwrite:
            del f[apogee_id]

        elif apogee_id in f and not overwrite:
            # should not get here!!
            raise RuntimeError("How did I get here???")

        g = f.create_group(apogee_id)

        g.create_dataset('p0', data=p0)
        g.create_dataset('chain', data=chain)

        # metadata
        g.attrs['n_walkers'] = n_walkers
        g.attrs['n_steps'] = n_steps
        g.attrs['n_burnin'] = n_burnin

    # plot orbits computed from the samples
    logger.debug("Plotting the MCMC samples...")

    fig,ax = plt.subplots(1,1,figsize=(8,6))

    fig = model.plot_rv_samples(chain[:,-1], ax=ax)
    _ = model.data.plot(ax=ax)
    fig.tight_layout()
    fig.savefig(join(PLOT_PATH, "{}-1-rv-curves.png".format(apogee_id)), dpi=256)

    # make a corner plot
    flatchain = np.vstack(chain[:,-256:])
    plot_pars = model.vec_to_plot_pars(flatchain)
    troup_vals = [np.log(troup_orbit.P.to(u.day).value), troup_orbit.mf.value,
                  troup_orbit.ecc, troup_orbit.omega.to(u.degree).value,
                  troup_orbit.t0.mjd, -troup_orbit.v0.to(u.km/u.s).value, 0.]
    fig = corner.corner(plot_pars, labels=model.plot_labels, truths=troup_vals)
    fig.tight_layout()
    fig.savefig(join(PLOT_PATH, "{}-2-corner.png".format(apogee_id)), dpi=256)
    logger.debug("done!")

    # make MCMC diagnostic plots as well (e.g., acceptance fraction, chain traces)
    plot_mcmc_diagnostics(sampler, p0, model, sampler_name, apogee_id)

    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true", help="Overwrite any existing data.")
    parser.add_argument("--seed", dest="seed", default=42, type=int,
                        help="Random number seed")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", dest="apogee_id", default=None,
                       type=str, help="APOGEE ID")
    group.add_argument("--index", dest="index", default=None, type=int,
                       help="Index of Troup target to run.")

    # MCMC
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--n-steps", dest="n_steps", default=4096,
                        type=int, help="Number of MCMC steps")
    parser.add_argument("--n-walkers", dest="n_walkers", default=256,
                        type=int, help="Number of MCMC walkers")
    parser.add_argument("--n-burnin", dest="n_burnin", default=128,
                        type=int, help="Number of MCMC burn-in steps")
    parser.add_argument("-s", "--sampler", dest="sampler_name", default="kombine",
                        type=str, help="Which MCMC sampler to use",
                        choices=["emcee", "kombine"])

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    np.random.seed(args.seed)

    main(args.apogee_id, index=args.index, n_walkers=args.n_walkers, n_steps=args.n_steps,
         mpi=args.mpi, n_burnin=args.n_burnin,
         sampler_name=args.sampler_name,
         overwrite=args.overwrite)
