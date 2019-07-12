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

# Prep. and write

f = h5py.File('fields3D.hdf5', 'w')

f0 = f.create_group('V_p')
f0.create_dataset('V_p', data=V_p)
f0.attrs['shape'] = shape
f0.attrs['extent'] = extent

f1 = f.create_group('Q_p')
f1.create_dataset('Q_p', data=Q_p)
f1.attrs['shape'] = shape
f1.attrs['extent'] = extent

f2 = f.create_group('V_s')
f2.create_dataset('V_s', data=V_s)
f2.attrs['shape'] = shape
f2.attrs['extent'] = extent

f3 = f.create_group('Q_s')
f3.create_dataset('Q_s', data=Q_s)
f3.attrs['shape'] = shape
f3.attrs['extent'] = extent

f4 = f.create_group('rho')
f4.create_dataset('rho', data=rho)
f4.attrs['shape'] = shape
f4.attrs['extent'] = extent

f.close()
