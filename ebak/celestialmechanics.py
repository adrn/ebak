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

def mean_anomaly_from_eccentric_anomaly(E, e):
    """
    # inputs:
    - E: eccentric anomaly (rad)
    - e: eccentricity

    # outputs:
    - M: mean anomaly (rad)
    """
    return E - e * np.sin(E)

def eccentric_anomaly_from_mean_anomaly(M, e, tol=1e-14, maxiter=100):
    """
    # inputs:
    - M: mean anomaly (rad)
    - e: eccentricity
    - tol: [read the source]
    - maxiter: [read the source]

    # outputs:
    - eccentric anomaly (rad)

    # bugs / issues:
    - MAGIC numbers 1e-14, 100
    - somewhat tested
    """
    iter = 0
    deltaM = np.Inf
    E = M + e * np.sin(M)
    while (iter < maxiter) and (abs(deltaM) > tol):
        deltaM = (M - mean_anomaly_from_eccentric_anomaly(E, e))
        E = E + deltaM / (1. - e * np.cos(E))
        iter += 1
    return E

def true_anomaly_from_eccentric_anomaly(E, e):
    """
    # inputs:
    - E: eccentric anomaly (rad)
    - e: eccentricity

    # outputs:
    - f: true anomaly (rad)

    # bugs / issues:
    - somewhat tested
    """
    cE, sE = np.cos(E), np.sin(E)
    f = np.arccos((cE - e) / (1.0 - e * cE))
    f *= (np.sign(np.sin(f)) * np.sign(sE))
    return f

def d_eccentric_anomaly_d_mean_anomaly(E, e):
    """
    # inputs:
    - E: eccentric anomaly (rad)
    - e: eccentricity

    # outputs:
    - dE / dM: derivative of one anomaly wrt the other

    # bugs / issues:
    - somewhat tested
    """
    return 1. / (1. - e * np.cos(E))

def d_true_anomaly_d_eccentric_anomaly(E, f, e):
    """
    # inputs:
    - E: eccentric anomaly (rad)
    - f: true anomaly (rad)
    - e: eccentricity

    # outputs:
    - df / dE: derivative of one anomaly wrt the other

    # bugs / issues:
    - insane assert
    - somewhat tested
    """
    cf, sf = np.cos(f), np.sin(f)
    cE, sE = np.cos(E), np.sin(E)
    assert np.allclose(cE, (e + cf) / (1. + e * cf))
    return (sE / sf) * (1. + e * cf) / (1. - e * cE)

def Z_from_elements(P, a, sini, e, omega, time, time0):
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
    - could be made more efficient (there are lots of re-dos of trig calls)
    - definitely something is wrong -- plots look wrong...!
    """
    dMdt = 2. * np.pi / P
    M = (time - time0) * dMdt
    E = eccentric_anomaly_from_mean_anomaly(M, e)
    f = true_anomaly_from_eccentric_anomaly(E, e)
    r = a * (1. - e * np.cos(E))
    return r * np.sin(omega + f) * sini

def rv_from_elements(P, a, sini, e, omega, time, time0):
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
    - rv: radial velocity (AU/d ?)

    # bugs / issues:
    - could be made more efficient (there are lots of re-dos of trig calls)
    - definitely something is wrong -- plots look wrong...!
    """
    dMdt = 2. * np.pi / P
    M = (time - time0) * dMdt
    E = eccentric_anomaly_from_mean_anomaly(M, e)
    f = true_anomaly_from_eccentric_anomaly(E, e)
    dEdt = d_eccentric_anomaly_d_mean_anomaly(E, e) * dMdt
    dfdt = d_true_anomaly_d_eccentric_anomaly(E, f, e) * dEdt
    r = a * (1. - e * np.cos(E))
    drdt = a * e * np.sin(E) * dEdt
    rv = r * np.cos(omega + f) * sini * dfdt + np.sin(omega + f) * sini * drdt
    return rv

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
        Z, Z1, Z2 = (Z_from_elements(P, a, sini, e, omega, tt, time0) for tt in threetimes)
        rv = rv_from_elements(P, a, sini, e, omega, time, time0)
        rv2 = (Z2 - Z1) / dt
        if np.abs(rv - rv2) > (a / P) * (1. / big):
            print("RV", rv, rv2, rv - rv2)
            assert False
    return True

if __name__ == "__main__":
    test_everything()
