from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.time as atime
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
import emcee
import corner

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
    p0 = model.get_par_vec()

    plot_pars = model.vec_to_plot_pars(p0)
    many_p0 = np.repeat(p0[None], 16, axis=0)
    many_plot_pars = model.vec_to_plot_pars(many_p0)

    for i in range(many_plot_pars.shape[0]):
        assert np.allclose(many_plot_pars, plot_pars)

# TODO: add decorator for slow test
def test_sample_from_prior():
    """
    Functional test: run with no data to sample from prior
    """
    n_walkers = 128
    n_steps = 2048

    no_data = RVData(np.array([]),
                     rv=[]*u.km/u.s,
                     ivar=[]/(u.km/u.s)**2)

    init_orbit = SimulatedRVOrbit(P=521.*u.day, a_sin_i=2*u.au, ecc=0.3523,
                                  omega=21.85*u.degree, phi0=-11.723*u.degree,
                                  v0=27.41*u.km/u.s)

    model = OrbitModel(data=no_data, orbit=init_orbit)
    assert np.isfinite(model.ln_posterior())

    p0 = emcee.utils.sample_ball(model.get_par_vec(),
                                 1E-6*model.get_par_vec() + 1E-6,
                                 size=n_walkers)
    sampler = emcee.EnsembleSampler(n_walkers, dim=p0.shape[1],
                                    lnpostfn=model)

    pos,_,_ = sampler.run_mcmc(p0, N=n_steps)

    # walker plots
    fig,axes = plt.subplots(p0.shape[1], 1, figsize=(5,18), sharex=True)
    for i in range(p0.shape[1]):
        axes[i].set_ylabel(model.vec_labels[i])
        axes[i].plot(sampler.chain[...,i].T, drawstyle='steps', alpha=0.1, marker=None)
    fig.tight_layout()

    # corner plot
    flatchain = np.vstack(sampler.chain[:,-256:])
    plot_pars = model.vec_to_plot_pars(flatchain.T).T
    fig = corner.corner(plot_pars, labels=model.plot_labels)

    plt.show()
