# Standard library
import os
from os.path import abspath, join, split, exists
import time
import sys

# Third-party
from astropy import log as logger
from astropy.io import fits
import astropy.table as tbl
import astropy.coordinates as coord
import astropy.units as u
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import corner
from gala.util import get_pool

# Project
from ebak.singleline.data import RVData
from ebak.sampler import tensor_vector_scalar, marginal_ln_likelihood
from ebak.units import usys

_basepath = split(abspath(join(__file__, "..")))[0]
if not exists(join("..", "scripts")):
    raise IOError("You must run this script from inside the scripts directory:\n{}"
                  .format(join(_basepath, "scripts")))

# HACKS: Hard-set paths
ALLVISIT_DATA_PATH = join(_basepath, "data", "allVisit-l30e.2.fits")
PLOT_PATH = join(_basepath, "plots", "ahw2016-magic")
CACHE_PATH = join(_basepath, "cache")
for PATH in [PLOT_PATH, CACHE_PATH]:
    if not exists(PATH):
        os.mkdir(PATH)

APOGEE_ID = "2M03080601+7950502"
# n_samples = 2**19
n_samples = 2**8 # HACK: temporary
P_min = 16. # day
P_max = 8192. # day

def marginal_ll_worker(task):
    nl_p, data = task
    ATA,p,chi2 = tensor_vector_scalar(nl_p, data)
    return marginal_ln_likelihood(ATA, chi2)

def get_good_samples(nonlinear_p, data, pool):
    tasks = [[nonlinear_p[i], data] for i in range(len(nonlinear_p))]
    results = pool.map(marginal_ll_worker, tasks)
    marg_ll = np.array(results)[:,0]

    uu = np.random.uniform(size=n_samples)
    good_samples_bool = uu < np.exp(marg_ll - marg_ll.max())
    good_samples = nonlinear_p[np.where(good_samples_bool)]
    n_good = len(good_samples)
    logger.info("{} good samples".format(n_good))

    return good_samples

def samples_to_orbital_params_worker(task):
    nl_p, data = task
    P, phi0, ecc, omega = nl_p

    ATA,p,_ = tensor_vector_scalar(nl_p, data)

    cov = np.linalg.inv(ATA)
    v0,asini = np.random.multivariate_normal(p, cov)

    if asini < 0:
        # logger.warning("Swapping asini")
        asini = np.abs(asini)
        omega += np.pi

    return [P, asini, ecc, omega, phi0, -v0] # TODO: sign of v0?

def samples_to_orbital_params(nonlinear_p, data, pool):
    tasks = [[nonlinear_p[i], data] for i in range(len(nonlinear_p))]
    orbit_pars = pool.map(samples_to_orbital_params_worker, tasks)
    return np.array(orbit_pars).T

def main(n_procs=0, mpi=False, seed=42, overwrite=False):

    output_filename = join(CACHE_PATH, "ahw2016-magic.h5")

    # MPI shite
    pool = get_pool(mpi=mpi, threads=n_procs)
    # need load-balancing - see: https://groups.google.com/forum/#!msg/mpi4py/OJG5eZ2f-Pg/EnhN06Ozg2oJ

    # load data from APOGEE data
    logger.debug("Reading data from Troup catalog and allVisit files...")
    all_data = RVData.from_apogee(ALLVISIT_DATA_PATH, apogee_id=APOGEE_ID)

    # a time grid to plot RV curves of the model - used way later
    t_grid = np.linspace(all_data._t.min()-50, all_data._t.max()+50, 1024)

    # TODO: add jitter below
    # s = 0.5 * u.km/u.s

    # sample from priors in nonlinear parameters
    P = np.exp(np.random.uniform(np.log(P_min), np.log(P_max), size=n_samples))
    phi0 = np.random.uniform(0, 2*np.pi, size=n_samples)
    # MAGIC NUMBERS below: Kipping et al. 2013 (MNRAS 434 L51)
    ecc = np.random.beta(a=0.867, b=3.03, size=n_samples)
    omega = np.random.uniform(0, 2*np.pi, size=n_samples)

    # pack the nonlinear parameters into an array
    nl_p = np.vstack((P, phi0, ecc, omega)).T
    # Note: the linear parameters are (v0, asini)

    logger.info("Number of prior samples: {}".format(n_samples))

    # TODO: we may want to "curate" which datapoints are saved...
    idx = np.random.permutation(len(all_data))
    for n_delete in range(len(all_data)):
        if n_delete == 0:
            data = all_data
        else:
            data = all_data[idx[:-n_delete]]
        logger.debug("Removing {}/{} data points".format(n_delete, len(all_data)))

        # see if we already did this:
        with h5py.File(output_filename, 'a') as f:
            if str(n_delete) in f and not overwrite:
                continue # skip if already did this one

            elif str(n_delete) in f and overwrite:
                del f[str(n_delete)]

        nl_samples = get_good_samples(nl_p, data, pool) # TODO: save?
        orbital_params = samples_to_orbital_params(nl_samples, data, pool)

        # save the orbital parameters out to a cache file
        par_names = ['P', 'asini', 'ecc', 'omega', 'phi0', 'v0']
        par_units = [usys['time'], usys['length'], None, usys['angle'],
                     usys['angle'], usys['length']/usys['time']]
        with h5py.File(output_filename, 'r+') as f:
            g = f.create_group(str(n_delete))

            for i,name,unit in zip(range(len(par_names)), par_names, par_units):
                g.create_dataset(name, data=orbital_params[i])
                g[name].attrs['unit'] = str(unit)

        # --------------------------------------------------------------------
        # make some plots, yo

        # plot samples
        fig = plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(2, 2)
        ax_rv = plt.subplot(gs[0,:])
        ax_lnP_e = plt.subplot(gs[1,0])
        ax_lnP_asini = plt.subplot(gs[1,1])

        _max_n = min(n_eff, 256)
        shit_fuck = min(n_eff, 1024)

        Q = 3. # HACK
        pt_alpha = min(0.8, max(0.2, 0.8 + 0.6*(np.log(2)-np.log(shit_fuck))/(np.log(1024)-np.log(2)))) # YEA BABY
        print(pt_alpha)
        for fucky,j in enumerate(good_samples[:shit_fuck]):
            ATA,p,_ = tensor_vector_scalar_kepler(nl_p[j], data)
            cov = np.linalg.inv(ATA)
            v0,asini = np.random.multivariate_normal(p, cov)

            P, phi0, ecc, omega = nl_p[j]

            if asini < 0:
                # logger.warning("Swapping asini")
                asini = np.abs(asini)
                omega += np.pi
            t0 = find_t0(phi0, P, EPOCH)

            model_rv = (rv_from_elements(_t, P, asini, ecc, omega, t0, -v0)*u.au/u.day).to(u.km/u.s)
            if fucky < _max_n: # HACK
                line_alpha = 0.1 + Q*0.65 / (fucky + Q)
                ax_rv.plot(_t, model_rv, linestyle='-', marker=None,
                           alpha=line_alpha, color='r')

            ax_lnP_e.plot(np.log(P), ecc, marker='.', color='k',
                          alpha=pt_alpha, ms=8)
            ax_lnP_asini.plot(np.log(P), np.log(asini), marker='.', color='k',
                              alpha=pt_alpha, ms=8)

            logger.debug("Parameters:")
            logger.debug("\t P={:.2f} day".format(P))
            logger.debug("\t phi0={:.3f} rad".format(phi0))
            logger.debug("\t e={:.3f}".format(ecc))
            logger.debug("\t omega={:.3f} rad".format(omega))
            logger.debug("\t v0={:.3f} km/s".format((v0*u.au/u.day).to(u.km/u.s).value))
            logger.debug("\t asini={:.3f} au".format(asini))

        ax_rv.errorbar(data._t, data.rv.to(u.km/u.s).value,
                       yerr=data.stddev.to(u.km/u.s).value,
                       marker='o', linestyle='none', color='k')
        ax_rv.set_xlim(_t.min()-25, _t.max()+25)
        _rv = all_data.rv.to(u.km/u.s).value
        ax_rv.set_ylim(np.median(_rv)-25, np.median(_rv)+25)
        ax_rv.set_xlabel('MJD')
        ax_rv.set_ylabel('RV [km s$^{-1}$]')

        ax_lnP_e.set_xlim(np.log(P_min), 6.) # HACK
        ax_lnP_e.set_ylim(-0.1, 1.)
        ax_lnP_e.set_xlabel(r'$\ln P$')
        ax_lnP_e.set_ylabel(r'$e$')

        ax_lnP_asini.set_xlim(np.log(P_min), 6.) # HACK
        ax_lnP_asini.set_ylim(-4, 0)
        ax_lnP_asini.set_xlabel(r'$\ln P$')
        ax_lnP_asini.set_ylabel(r'$\ln (a \sin i)$')

        fig.tight_layout()
        fig.savefig(join(PLOT_PATH, 'leave_out_{}.png'.format(leave_out)))

        # fig = corner.corner(np.hstack((np.log(nl_p[:,0:1]), nl_p[:,1:])),
        #                     labels=['$\ln P$', r'$\phi_0$', '$e$', r'$\omega$'])
        # plt.savefig(join(PLOT_PATH, 'corner-leave_out_{}.png'.format(leave_out)))
        plt.close('all')

    # pool.close()

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

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    np.random.seed(args.seed)

    main(n_procs=args.n_procs, mpi=args.mpi, seed=args.seed,
         overwrite=args.overwrite)
