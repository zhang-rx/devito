import numpy as np

from conftest import skipif
from devito import Grid, TimeFunction, Eq, Operator, switchconfig
from devito.ir.iet import retrieve_iteration_tree

pytestmark = skipif(['yask', 'ops'])


class TestOffloading(object):

    @switchconfig(platform='nvidiaX')
    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), dle=('advanced', {'openmp': True}))

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        assert op.body[1].header[1].value ==\
            ('omp target enter data map(to: u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body[1].footer[0].value ==\
            ('omp target exit data map(from: u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')

    @switchconfig(platform='nvidiaX')
    def test_multiple_eqns(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        op = Operator([Eq(u.forward, u + v + 1), Eq(v.forward, u + v + 4)],
                      dle=('advanced', {'openmp': True}))

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        for i, f in enumerate([u, v]):
            assert op.body[2].header[2 + i].value ==\
                ('omp target enter data map(to: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})
            assert op.body[2].footer[i].value ==\
                ('omp target exit data map(from: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})

    @switchconfig(platform='nvidiaX')
    def test_op_apply(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid, dtype=np.int32)

        op = Operator(Eq(u.forward, u + 1), dle=('advanced', {'openmp': True}))

        time_steps = 1000
        op.apply(time_M=time_steps)

        assert np.all(np.array(u.data[0, :, :, :]) == time_steps)

    @switchconfig(platform='nvidiaX')
    def test_iso_ac(self):
        from examples.seismic import Model, TimeAxis, RickerSource, Receiver
        from devito import TimeFunction, solve, norm

        shape = (101, 101)
        spacing = (10., 10.)
        origin = (0., 0.)

        v = np.empty(shape, dtype=np.float32)
        v[:, :51] = 1.5
        v[:, 51:] = 2.5

        model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                      space_order=2, nbl=10)

        t0 = 0.
        tn = 1000.
        dt = model.critical_dt

        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        f0 = 0.010
        src = RickerSource(name='src', grid=model.grid, f0=f0,
                           npoint=1, time_range=time_range)

        src.coordinates.data[0, :] = np.array(model.domain_size) * .5
        src.coordinates.data[0, -1] = 20.

        rec = Receiver(name='rec', grid=model.grid, npoint=101, time_range=time_range)
        rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
        rec.coordinates.data[:, 1] = 20.

        u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)

        pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
        stencil = Eq(u.forward, solve(pde, u.forward))

        src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)

        rec_term = rec.interpolate(expr=u.forward)

        op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map,
                      dle=('advanced', {'openmp': True}))
        op(time=time_range.num-1, dt=model.critical_dt)

        assert np.isclose(norm(rec), 447.28362, atol=1e-3, rtol=0)
