from __future__ import division, print_function

__author__ = "David W. Hogg <david.hogg@nyu.edu>"

import warnings

# Third-party
import astropy.time as at
import astropy.units as u
import numpy as np
import pytest

from .celestialmechanics import *

def test_everything():
    np.random.seed(42)
    tt0 = 0. # d
    tt1 = 400. # d
    for n in range(256):
        P = np.exp(np.log(10.) + np.log(400./10.) * np.random.uniform()) # d
        a = (P / 300.) ** (2. / 3.) # AU
        e = np.random.uniform()
        omega = 2. * np.pi * np.random.uniform() # rad
        time0, time = tt0 + (tt1 - tt0) * np.random.uniform(size=2) # d
        sini = 1.0
        print("testing", P, a*sini, e, omega, time, time0)
        big = 65536.0 # MAGIC
        dt = P / big # d ; MAGIC
        dMdt = 2. * np.pi / P # rad / d
        threetimes = [time, time - 0.5 * dt, time + 0.5 * dt]
        M, M1, M2 = ((t - time0) * dMdt for t in threetimes)
        E, E1, E2 = (eccentric_anomaly_from_mean_anomaly(MM, e) for MM in [M, M1, M2])
        dEdM = d_eccentric_anomaly_d_mean_anomaly(E, e)
        dEdM2 = (E2 - E1) / (M2 - M1)
        if np.abs(dEdM - dEdM2) > (1. / big):
            print("dEdM", dEdM, dEdM2, dEdM - dEdM2)
            assert False
        f, f1, f2 = (true_anomaly_from_eccentric_anomaly(EE, e) for EE in [E, E1, E2])
        dfdE = d_true_anomaly_d_eccentric_anomaly(E, f, e)
        dfdE2 = (f2 - f1) / (E2 - E1)
        if np.abs(dfdE - dfdE2) > (1. / big):
            print("dfdE", dfdE, dfdE2, dfdE - dfdE2)
            assert False
        Z, Z1, Z2 = Z_from_elements(threetimes, P, a*sini, e, omega, time0)
        rv = rv_from_elements(time, P, a*sini, e, omega, time0, 0.)
        rv2 = (Z2 - Z1) / dt
        if np.abs(rv - rv2) > (a / P) * (1. / big):
            print("RV", rv, rv2, rv - rv2)
            assert False

# def test_functional():
#     import emcee
#     from . import SimulatedRVOrbit
#     from .singleline import RVData, OrbitModel
#     from .units import usys

#     data = RVData(t=[55933.51268,  55966.44446,  56052.30926,  55967.44955,
#                     56077.13696,  56081.21252,  55725.14849,  55726.14384,  55727.14015],
#                   rv=[0.00459471, 0.0050067 , 0.00460107, 0.00493878, 0.00475608,
#                       0.00485175, 0.00512318, 0.00526451, 0.005195]*u.au/u.day,
#                   ivar=[1.13864192e+09,  9.47996672e+08,  8.22171264e+08,
#                         1.23196275e+09,  7.20272064e+08,  2.96464576e+08,
#                         1.60310362e+09,  1.39964493e+09,  1.40657178e+09]*u.d**2/u.au**2)
#     orbit = SimulatedRVOrbit(P=14.464829849379742*u.day,
#                              a_sin_i=0.000865328820692887*u.au,
#                              omega=0.716177954118933*u.rad,
#                              ecc=0.51551958061843572,
#                              phi0=-0.67617184737855385*u.radian,
#                              v0=-0.004815773387206876*u.au/u.day)
#     model = OrbitModel(data=data, orbit=orbit)

#     n_steps = 128
#     n_walkers = 256
#     p0 = emcee.utils.sample_ball(model.get_par_vec(),
#                                  1E-3*model.get_par_vec(),
#                                  size=n_walkers)

#     # special treatment for ln_P
#     p0[:,0] = np.random.normal(np.log(model.orbit._P), 0.1, size=p0.shape[0])

#     # special treatment for s
#     p0[:,6] = np.abs(np.random.normal(0, 1E-3, size=p0.shape[0]) * u.km/u.s).decompose(usys).value

#     with warnings.catch_warnings():
#         warnings.filterwarnings('error')

#         for pp in p0:
#             try:
#                 assert np.isfinite(model(pp))
#             except RuntimeWarning:
#                 print("maxiter warning!")
#                 print(pp)
#                 break

#     _model = model.from_vec(pp)
#     print(_model.orbit.K.to(u.km/u.s), _model.orbit.ecc, _model.orbit.m_f)
