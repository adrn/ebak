from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np

# Project
from .. import usys
from ..celestialmechanics_class import SimulatedRVOrbit

__all__ = ['OrbitModel']

# Magic numbers:
_ivar_disk = (1 / (30.*u.km/u.s)**2).decompose(usys).value # used in ln_prior
jitter_scale = (0.3*u.km/u.s).decompose(usys).value # typical vel err.
EPOCH = 55555. # used for un-modding phi0

class OrbitModel(object):
    """
    Parameters
    ----------
    orbit : RVOrbit
    data : RVData

    """
    def __init__(self, data=None, orbit=None, s=0.*u.km/u.s):
        self.data = data

        if orbit is None:
            orbit = SimulatedRVOrbit(np.nan*u.yr, np.nan*u.au, np.nan,
                                     np.nan*u.radian, np.nan, v0=np.nan*u.km/u.s)
        self.orbit = orbit

        self._s = s.decompose(usys).value

    @property
    def s(self):
        return self._s*usys['length']/usys['time']

    def ln_likelihood(self):
        rvs = self.orbit._generate_rv_curve(self.data._t)
        new_ivar = self.data._ivar / (1 + self._s**2 * self.data._ivar)
        chi2 = new_ivar * (self.data._rv - rvs)**2

        return -0.5*chi2 + 0.5*np.log(new_ivar)

    def ln_prior(self):
        lnp = 0.

        # Jitter: for velocity error model
        if self._s < 0.:
            return -np.inf
        lnp += -self._s / jitter_scale

        # Mass function: log-normal centered on ln(3)
        m_f = self.orbit._m_f
        lnp += -0.5 * (np.log(m_f) - np.log(3.))**2 / (2.)**2
        if m_f < 0:
            return -np.inf

        # Orbital period: assumes sampler is stepping in log(P)
        if self.orbit._P < 0.1 or self.orbit._P > 8192*365.: # days
            return -np.inf

        # Semi-major axis and inclination
        if 1E-6 < self.orbit._a_sin_i < 16384.: # au
            lnp += -np.log(self.orbit._a_sin_i)
        else:
            return -np.inf

        # Eccentricity
        if self.orbit.ecc < 0. or self.orbit.ecc > 1.:
            return -np.inf

        # Phase0: uniform in phase0
        # HACK: closest t0 to 55555. at phase phi0
        _t0_epoch = 55000.
        if (self.orbit._t0 < (_t0_epoch-1.5*self.orbit._P) or
            self.orbit._t0 > (_t0_epoch+1.5*self.orbit._P)):
            return -np.inf

        # Systemic velocity: Gaussian with velocity dispersion of the disk
        lnp += -0.5 * _ivar_disk * self.orbit._v0**2

        return lnp

    def ln_posterior(self):
        lnp = self.ln_prior()
        if not np.isfinite(lnp):
            return -np.inf
        return lnp + self.ln_likelihood().sum()

    def __call__(self, p):
        self.set_par_from_vec(p)
        return self.ln_posterior()

    def from_vec(self, p):

        # TODO: fix this

        _P = np.exp(ln_P)
        _asini = np.sqrt(asini_cos_phi0**2 + asini_sin_phi0**2)
        _phi0 = np.arctan2(asini_sin_phi0, asini_cos_phi0)
        # TODO: there should be a ._phi0 on Orbit

        ecc = sqrte_cos_pomega**2 + sqrte_sin_pomega**2
        _omega = np.arctan2(sqrte_sin_pomega, sqrte_cos_pomega)

        orbit = SimulatedRVOrbit(P=_P*usys['time'],
                                 a_sin_i=_asini*usys['length'],
                                 ecc=ecc, omega=_omega*usys['angle'],
                                 t0=_t0, v0=_v0*usys['length']/usys['time'])

        # nuisance parameters
        s = _s*usys['length']/usys['time']

        return self.__class__(data=self.data, orbit=orbit, s=s)

    def set_par_from_vec(self, p):
        (ln_P,
         asini_cos_phi0, asini_sin_phi0,
         sqrte_cos_pomega, sqrte_sin_pomega,
         _v0, _s) = p

        self.orbit._P = np.exp(ln_P)
        self.orbit._a_sin_i = np.sqrt(asini_cos_phi0**2 + asini_sin_phi0**2)
        self.orbit.ecc = sqrte_cos_pomega**2 + sqrte_sin_pomega**2
        self.orbit._omega = np.arctan2(sqrte_sin_pomega, sqrte_cos_pomega)
        self.orbit._phi0 = _np.arctan2(asini_sin_phi0, asini_cos_phi0)
        self.orbit._v0 = _v0

        # nuisance parameters
        self._s = _s

    def get_par_vec(self):
        _phi0 = 2*np.pi*self.orbit._t0 / self.orbit._P
        return np.array([np.log(self.orbit._P),
                         self.orbit._a_sin_i*np.cos(_phi0),
                         self.orbit._a_sin_i*np.sin(_phi0),
                         np.sqrt(self.orbit.ecc)*np.cos(self.orbit._omega),
                         np.sqrt(self.orbit.ecc)*np.sin(self.orbit._omega),
                         self.orbit._v0,
                         self._s])

    def vec_to_plot_pars(self, p):
        """
        Convert an MCMC parameter vector to the parameters we will plot.
        """

        p = np.atleast_1d(p)
        if p.ndim > 2:
            raise ValueError("Shape of input p must be <= 2 (e.g., pass flatchain")

        model = self.from_vec(p)
        orbit = model.orbit

        return np.vstack((np.log(orbit.P.to(u.day).value),
                          orbit.m_f.to(u.Msun).value,
                          orbit.ecc,
                          orbit.omega.to(u.degree).value,
                          orbit.t0.mjd,
                          orbit.v0.to(u.km/u.s).value,
                          model.s.to(u.km/u.s).value))

    # copy methods
    def __copy__(self):
        return self.__class__(data=self.data.copy(),
                              orbit=self.orbit.copy(),
                              s=self.s.copy())
    def copy(self):
        return self.__copy__()
