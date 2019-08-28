import h5py

from mpi4py import MPI
import numpy as np

from devito import Grid, Function
from devito import configuration
from devito.tools import as_tuple

from devito.data import Decomposition
from devito.mpi import Distributor

configuration['mpi'] = True

''' These need to be from a file or input manually '''

shape = (201, 101, 101)
extent = (200., 100., 100.) # Input manually/from stored file?

''' Fields '''

V_p = np.fromfile('vp.dat')
Q_p = np.fromfile('qp.dat')
V_s = np.fromfile('vs.dat')
Q_s = np.fromfile('qs.dat')
rho = np.fromfile('rho.dat')

V_p = V_p.reshape(shape)
Q_p = Q_p.reshape(shape)
V_s = V_s.reshape(shape)
Q_p = Q_p.reshape(shape)
rho = rho.reshape(shape)

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
