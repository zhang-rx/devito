import h5py

from mpi4py import MPI
import numpy as np

from devito import Grid, Function
from devito import configuration
from devito.tools import as_tuple

from devito.data import Decomposition
from devito.mpi import Distributor

configuration['mpi'] = True

shape = (201, 101, 101)
extent = (200., 100., 100.)

''' Fields '''

V_p = np.zeros(shape)
Q_p = np.zeros(shape)
V_s = np.zeros(shape)
Q_s = np.zeros(shape)
rho = np.zeros(shape)

for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            if k < 50:
                V_p[i, j, k] = 1.52
                Q_p[i, j, k] = 10000.
                V_s[i, j, k] = 0.
                Q_s[i, j, k] = 0.
                rho[i, j, k] = 1.05
            elif 50 <= k < 54:
                V_p[i, j, k] = 1.6
                Q_p[i, j, k] = 40.
                V_s[i, j, k] = 0.4
                Q_s[i, j, k] = 30.
                rho[i, j, k] = 1.3            
            else:
                V_p[i, j, k] = 2.2
                Q_p[i, j, k] = 100.0
                V_s[i, j, k] = 1.2
                Q_s[i, j, k] = 70.
                rho[i, j, k] = 2.0

V_p.tofile('vp.dat')
Q_p.tofile('qp.dat')
V_s.tofile('vs.dat')
Q_s.tofile('qs.dat')
rho.tofile('rho.dat')
