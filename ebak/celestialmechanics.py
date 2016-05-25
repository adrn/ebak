"""
This file is part of the EBAK project.
Copyright 2016 David W. Hogg (NYU).

# Celestial mechanics for the EBAK project

## Bugs / issues:
- Should I permit inputs of sin and cos instead of just angles?
- Totally untested
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

def eccentric_anomaly_from_mean_anomaly(M, e, tol=1e-8, maxiter=100)
    """
    # inputs:
    - M: mean anomaly (rad)
    - e: eccentricity
    - tol: [read the source]
    - maxiter: [read the source]

    # outputs:
    - eccentric anomaly (rad)

    # bugs:
    - MAGIC numbers 1e-8, 100
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
    """
    return 1. / (1. - e * np.cos(E))

def d_true_anomaly_d_eccentric_anomaly(E, f, e):
    """
    # inputs:n
    - E: eccentric anomaly (rad)
    - f: true anomaly (rad)
    - e: eccentricity

    # outputs:
    - df / dE: derivative of one anomaly wrt the other

    # issues:
    - insane assert
    """
    cf, sf = np.cos(f), np.sin(f)
    cE, sE = np.cos(E), np.sin(E)
    assert np.close(cE, (e + cf) / (1. + e * cf))
    return (sE / sf) * (1. - e * e) / (1. + e * cf) ** 2
