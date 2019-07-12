import h5py

#from mpi4py import MPI
import numpy as np
import sympy as sp

from devito import *
from examples.seismic.source import WaveletSource, RickerSource, GaborSource, TimeAxis
from examples.seismic import plot_image

from devito.tools import as_tuple

#configuration['mpi'] = True

# Read file
fname = 'fields2D.hdf5'
f = h5py.File(fname, 'r')

vp = f[list(f['V_p'].keys())[0]]
qp = f[list(f['Q_p'].keys())[0]]
vs = f[list(f['V_s'].keys())[0]]
qs = f[list(f['Q_s'].keys())[0]]
de = f[list(f['rho'].keys())[0]]

shape = as_tuple(vp.attrs['shape'])
extent = as_tuple(vp.attrs['extent'])

# Create grid
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, z))

# Fields to read from file
V_p = Function(name='V_p', grid=grid)
Q_p = Function(name='Q_p', grid=grid)
V_s = Function(name='V_s', grid=grid)
Q_s = Function(name='Q_s', grid=grid)
rho = Function(name='rho', grid=grid)

# Read data from file into fields
lslice = []
for i in V_p._decomposition:
    lslice.append(slice(i.loc_abs_min, i.loc_abs_max+1, 1))
lslice = as_tuple(lslice)

# Fill functions
vpkey = list(vp.keys())[0]
V_p.data[lslice] = np.array(vp[vpkey][lslice])

qpkey = list(qp.keys())[0]
Q_p.data[lslice] = np.array(qp[qpkey][lslice])

vskey = list(vs.keys())[0]
V_s.data[lslice] = np.array(vs[vskey][lslice])

qskey = list(qs.keys())[0]
Q_s.data[lslice] = np.array(qs[qskey][lslice])

dekey = list(de.keys())[0]
rho.data[lslice] = np.array(de[dekey][lslice])

f.close()

# Source freq. in MHz
f0 = 0.12
# Thorbecke's parameter notation
cp2 = V_p*V_p
cs2 = V_s*V_s
ro = 1./rho

mu = cs2*ro
l = (cp2*ro - 2*mu)
k = 1.0/(l + 2*mu)
pi = l + 2*mu

t_s = (sp.sqrt(1.+1./Q_p**2)-1./Q_p)/f0
t_ep = 1./(f0**2*t_s)
t_es = (1.+f0*Q_s*t_s)/(f0*Q_s-f0**2*t_s)

# NOTE: max(Q_p) should be stored in data file
Qw = 10000.
t_sM = (np.sqrt(1.+1./Qw**2)-1./Qw)/f0

# Time steps
t0, tn = 0., 50.
dt = 0.01*t_sM
time_range = TimeAxis(start=t0, stop=tn, step=dt)

# PDE fn's
so = 2
vx= TimeFunction(name='vx', grid=grid, staggered=x, space_order=so)
vz = TimeFunction(name='vz', grid=grid, staggered=z, space_order=so)
sxx = TimeFunction(name='sxx', grid=grid, staggered=NODE, space_order=so)
szz = TimeFunction(name='szz', grid=grid, staggered=NODE, space_order=so)
sxz = TimeFunction(name='sxz', grid=grid, staggered=(x, z), space_order=so)
rxx = TimeFunction(name='rxx', grid=grid, staggered=NODE, space_order=so)
rzz = TimeFunction(name='rzz', grid=grid, staggered=NODE, space_order=so)
rxz = TimeFunction(name='rxz', grid=grid, staggered=(x, z), space_order=so)

t = grid.stepping_dim
time = grid.time_dim

# Source

src = RickerSource(name='src', grid=grid, f0=f0, time_range=time_range)
src.coordinates.data[:] = np.array([100., 35.])

# The source injection term
src_xx = src.inject(field=sxx.forward, expr=src)
src_zz = src.inject(field=szz.forward, expr=src)

# fdelmodc reference implementation
eq_vx = Eq(vx.forward, vx + dt*ro*(sxx.dx + sxz.dz))

eq_vz = Eq(vz.forward, vz + ro*dt*(sxz.dx + szz.dz))

eq_sxx = Eq(sxx.forward, sxx + dt*pi*t_ep/t_s*(vx.forward.dx+vz.forward.dz) \
        - 2.*dt*mu*t_es/t_s*vz.forward.dz + dt*rxx.forward)
eq_szz = Eq(szz.forward, szz + dt*pi*t_ep/t_s*(vx.forward.dx+vz.forward.dz) \
        - 2.*dt*mu*t_es/t_s*vx.forward.dx + dt*rzz.forward)
eq_sxz = Eq(sxz.forward, sxz + dt*mu*t_es/t_s*(vx.forward.dz+vz.forward.dx) + dt*rxz.forward)

eq_rxx = Eq(rxx.forward, rxx-dt*1./t_s*(rxx+pi*(t_ep/t_s-1)*(vx.forward.dx+vz.forward.dz) \
         - 2*mu*(t_es/t_s-1)*vz.forward.dz))
eq_rzz = Eq(rzz.forward, rzz-dt*1./t_s*(rzz+pi*(t_ep/t_s-1)*(vx.forward.dx+vz.forward.dz) \
         - 2*mu*(t_es/t_s-1)*vx.forward.dx))
eq_rxz = Eq(rxz.forward, rxz - dt*1/t_s*(rxz+mu*(t_es/t_s-1)*(vx.forward.dz+vz.forward.dx)))

op = Operator([eq_vx, eq_vz, eq_rxx, eq_rzz, eq_rxz, eq_sxx, eq_szz, eq_sxz] + src_xx + src_zz)

# Reset the fields
vx.data[:] = 0.
vz.data[:] = 0.
sxx.data[:] = 0.
szz.data[:] = 0.
sxz.data[:] = 0.
rxx.data[:] = 0.
rzz.data[:] = 0.
rxz.data[:] = 0.

op()

plot_image(vx.data[0], cmap="seismic")
#plot_image(vz.data[0], cmap="seismic")
#plot_image(sxx.data[0], cmap="seismic")
#plot_image(szz.data[0], cmap="seismic")
#plot_image(sxz.data[0], cmap="seismic")
#plot_image(rxx.data[0], cmap="seismic")
#plot_image(rzz.data[0], cmap="seismic")
#plot_image(rxz.data[0], cmap="seismic")

from IPython import embed; embed()
