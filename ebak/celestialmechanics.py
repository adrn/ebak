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
input numpy as np

def mean_anomaly_from_eccentric_anomaly(E, e):
    """
    # inputs:
    - E: eccentric anomaly (rad)
    - e: eccentricity

    # outputs:
    - M: mean anomaly (rad)
    """
    return E - e * np.sin(E)

def eccentric_anomaly_from_mean_anomaly(M, e, tol=1e-8, maxiter=100):
    """
    # inputs:
    - M: mean anomaly (rad)
    - e: eccentricity
    - tol: [read the source]
    - maxiter: [read the source]

    # outputs:
    - eccentric anomaly (rad)

    # bugs / issues:
    - MAGIC numbers 1e-8, 100
    - totally untested
    """
    iteration = 0
    deltaM = np.Inf
    E = M + e * np.sin(M)
    while (iteration < maxiter) and (abs(deltaM) > tol):
        deltaM = (M - mean_anomaly_from_eccentric_anomaly(E, e))
        E = E + deltaM / (1. - e * cos(E))
        iteration += 1
    return E

def true_anomaly_from_eccentric_anomaly(E, e):
    """
    # inputs:
    - E: eccentric anomaly (rad)
    - e: eccentricity

    # outputs:
    - f: true anomaly (rad)

    # bugs / issues:
    - totally untested
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
    - totally untested
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
    """
    cf, sf = np.cos(f), np.sin(f)
    cE, sE = np.cos(E), np.sin(E)
    assert np.close(cE, (e + cf) / (1. + e * cf))
    return (sE / sf) * (1. - e * e) / (1. + e * cf) ** 2

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
    - totally untested
    """
    dMdt = 2. * np.pi / P
    M = np.mod((time - time0) * dMdt, 2. * np.pi)
    E = eccentric_anomaly_from_mean_anomaly(M, e)
    f = true_anomaly_from_eccentric_anomaly(E, e)
    dEdt = d_eccentric_anomaly_d_mean_anomaly(E, e) * dMdt
    dfdt = d_true_anomaly_d_eccentric_anomaly(E, f, e) * dEdt
    r = a * (1. - e * e) / (1 + e * np.cos(f))
    rv = r * np.cos(omega + f) * sini * dfdt
    return rv
