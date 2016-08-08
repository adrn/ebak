"""
This file is part of the EBAK project.
Copyright 2016 David W. Hogg (NYU).

# Celestial mechanics for the EBAK project

## comments:
- parameterization from Winn http://arxiv.org/abs/1001.2010
- mean, eccentric, and true anomaly formulae from Wikipedia https://en.wikipedia.org/wiki/Eccentric_anomaly

## bugs / issues:
- should I permit inputs of sin and cos instead of just angles?
- totally untested
"""
import numpy as np

def mean_anomaly_from_eccentric_anomaly(Es, e):
    """
    # inputs:
    - Es: eccentric anomalies (rad)
    - e: eccentricity

    # outputs:
    - Ms: mean anomalies (rad)
    """
    return Es - e * np.sin(Es)

def eccentric_anomaly_from_mean_anomaly(Ms, e, tol=1.e-14, maxiter=100):
    """
    # inputs:
    - Ms: mean anomaly (rad)
    - e: eccentricity
    - tol: [read the source]
    - maxiter: [read the source]

    # outputs:
    - Es: eccentric anomaly (rad)

    # bugs / issues:
    - MAGIC numbers 1e-14, 100
    - somewhat tested
    - could be parallelized
    """
    iter = 0
    deltaMs = np.Inf
    Es = Ms + e * np.sin(Ms)
    while (iter < maxiter) and np.any(np.abs(deltaMs) > tol):
        deltaMs = (Ms - mean_anomaly_from_eccentric_anomaly(Es, e))
        Es = Es + deltaMs / (1. - e * np.cos(Es))
        iter += 1
    return Es

def true_anomaly_from_eccentric_anomaly(Es, e):
    """
    # inputs:
    - Es: eccentric anomaly (rad)
    - e: eccentricity

    # outputs:
    - fs: true anomaly (rad)

    # bugs / issues:
    - somewhat tested
    """
    cEs, sEs = np.cos(Es), np.sin(Es)
    fs = np.arccos((cEs - e) / (1.0 - e * cEs))
    fs *= (np.sign(np.sin(fs)) * np.sign(sEs))
    return fs

def d_eccentric_anomaly_d_mean_anomaly(Es, e):
    """
    # inputs:
    - Es: eccentric anomaly (rad)
    - e: eccentricity

    # outputs:
    - dE / dM: derivatives of one anomaly wrt the other

    # bugs / issues:
    - somewhat tested
    """
    return 1. / (1. - e * np.cos(Es))

def d_true_anomaly_d_eccentric_anomaly(Es, fs, e):
    """
    # inputs:
    - Es: eccentric anomaly (rad)
    - fs: true anomaly (rad)
    - e: eccentricity

    # outputs:
    - df / dE: derivative of one anomaly wrt the other

    # bugs / issues:
    - insane assert
    - somewhat tested
    """
    cfs, sfs = np.cos(fs), np.sin(fs)
    cEs, sEs = np.cos(Es), np.sin(Es)
    assert np.allclose(cEs, (e + cfs) / (1. + e * cfs))
    return (sEs / sfs) * (1. + e * cfs) / (1. - e * cEs)

def Z_from_elements(times, P, a, sini, e, omega, time0):
    """
    # inputs:
    - P: period (d)
    - a: semi-major axis for star from system barycenter (will be negative for one of the stars?) (AU ?)
    - sini: sine of the inclination
    - e: eccentricity
    - omega: perihelion argument parameter from Winn
    - time: BJD of observation (d)
    - time0: time of "zeroth" pericenter (d)

    # outputs:
    - Z: line-of-sight position (AU)

    # bugs / issues:
    - doesn't include system Z value (Z offset or Z zeropoint)
    - could be made more efficient (there are lots of re-dos of trig calls)
    - definitely something is wrong -- plots look wrong...!
    """
    dMdt = 2. * np.pi / P
    Ms = (times - time0) * dMdt
    Es = eccentric_anomaly_from_mean_anomaly(Ms, e)
    fs = true_anomaly_from_eccentric_anomaly(Es, e)
    rs = a * (1. - e * np.cos(Es))
    return rs * np.sin(omega + fs) * sini

def rv_from_elements(times, P, a, sini, e, omega, time0):
    """
    # inputs:
    - times: BJD of observations (d)
    - P: period (d)
    - a: semi-major axis for star from system barycenter (will be negative for one of the stars?) (AU ?)
    - sini: sine of the inclination
    - e: eccentricity
    - omega: perihelion argument parameter from Winn
    - time0: time of "zeroth" pericenter (d)

    # outputs:
    - rv: radial velocity (AU/d ?)

    # bugs / issues:
    - doesn't include velocity zeropoint
    - could be made more efficient (there are lots of re-dos of trig calls)
    - definitely something is wrong -- plots look wrong...!
    """
    dMdt = 2. * np.pi / P
    Ms = (times - time0) * dMdt
    Es = eccentric_anomaly_from_mean_anomaly(Ms, e)
    fs = true_anomaly_from_eccentric_anomaly(Es, e)
    dEdts = d_eccentric_anomaly_d_mean_anomaly(Es, e) * dMdt
    dfdts = d_true_anomaly_d_eccentric_anomaly(Es, fs, e) * dEdts
    rs = a * (1. - e * np.cos(Es))
    drdts = a * e * np.sin(Es) * dEdts
    rvs = rs * np.cos(omega + fs) * sini * dfdts + np.sin(omega + fs) * sini * drdts
    return rvs

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
        print("testing", P, a, sini, e, omega, time, time0)
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
        Z, Z1, Z2 = Z_from_elements(threetimes, P, a, sini, e, omega, time0)
        rv = rv_from_elements(time, P, a, sini, e, omega, time0)
        rv2 = (Z2 - Z1) / dt
        if np.abs(rv - rv2) > (a / P) * (1. / big):
            print("RV", rv, rv2, rv - rv2)
            assert False
    return True

if __name__ == "__main__":
    test_everything()
