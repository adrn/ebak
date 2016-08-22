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

# Alternate pool implementation
from mpi4py import MPI
class MPIPool(object):

    def __init__(self, comm, master=0):
        assert comm.size > 1
        assert 0 <= master < comm.size
        self.comm = comm
        self.master = master
        self.workers = set(range(comm.size))
        self.workers.discard(self.master)

    def is_master(self):
        return self.master == self.comm.rank

    def is_worker(self):
        return self.comm.rank in self.workers

    def map(self, function, iterable):
        assert self.is_master()

        comm = self.comm
        workerset = self.workers.copy()
        tasklist = [(tid, (function, arg)) for tid, arg in enumerate(iterable)]
        resultlist = [None] * len(tasklist)
        pending = len(tasklist)

        while pending:

            if workerset and tasklist:
                worker = workerset.pop()
                taskid, task = tasklist.pop()
                comm.send(task, dest=worker, tag=taskid)

            if tasklist:
                flag = comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                if not flag: continue
            else:
                comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

            status = MPI.Status()
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.source
            workerset.add(worker)
            taskid = status.tag
            resultlist[taskid] = result
            pending -= 1

        return resultlist

    def start(self):
        if not self.is_worker(): return
        comm = self.comm
        master = self.master
        status = MPI.Status()
        while True:
            task = comm.recv(source=master, tag=MPI.ANY_TAG, status=status)
            if task is None: break
            function, arg = task
            result = function(arg)
            comm.ssend(result, master, status.tag)

    def stop(self):
        if not self.is_master(): return
        for worker in self.workers:
            self.comm.send(None, worker, 0)

class Gauss(object):
    def __init__(self, cov):
        self.cov = cov
        self.ndim = self.cov.shape[0]

    def logpdf(self, x):
        return mvn.logpdf(x, mean=np.zeros(self.ndim), cov=self.cov)

    def __call__(self, x):
        return self.logpdf(x)

def main(option, n_walkers):
    n_dim = 8

    # Option 1: mpipool
    if option == 1:
        pool = get_pool(mpi=True, loadbalance=True)

        # set up target probability distribution
        A = np.random.rand(n_dim, n_dim)
        cov = A*A.T + n_dim*np.eye(n_dim);
        ln_prob = Gauss(cov)

        # sampling
        sampler = kombine.Sampler(n_walkers, n_dim, ln_prob, pool=pool)

        p0 = np.random.uniform(-10, 10, size=(n_walkers, n_dim))
        p, post, q = sampler.burnin(p0)
        sampler.run_mcmc(16)

        pool.close()

    # -------------------------------------------------------------------

    # Option 2: pool implementation above
    elif option == 2:
        pool = MPIPool(MPI.COMM_WORLD)
        pool.start()

        if pool.is_master():
            # set up target probability distribution
            A = np.random.rand(n_dim, n_dim)
            cov = A*A.T + n_dim*np.eye(n_dim);
            ln_prob = Gauss(cov)

            # sampling
            sampler = kombine.Sampler(n_walkers, n_dim, ln_prob, pool=pool)

            p0 = np.random.uniform(-10, 10, size=(n_walkers, n_dim))
            p, post, q = sampler.burnin(p0)
            sampler.run_mcmc(16)

        pool.stop()

    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("--option", dest="option", required=True, type=int)
    parser.add_argument("--n-walkers", dest="n_walkers", default=256,
                        type=int, help="Number of MCMC walkers")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    logger.setLevel(logging.DEBUG)

    main(args.option, n_walkers=args.n_walkers)
