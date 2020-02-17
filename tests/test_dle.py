from functools import reduce
from operator import mul

import numpy as np
import pytest
from unittest.mock import patch
from sympy import cos

from conftest import skipif
from devito import (Grid, Dimension, Function, TimeFunction, SparseTimeFunction,
                    SubDimension, Eq, Inc, Operator, switchconfig)
from devito.exceptions import InvalidArgument
from devito.ir.iet import Call, Iteration, Conditional, FindNodes, retrieve_iteration_tree
from devito.passes import BlockDimension, NThreads, NThreadsNonaffine
from devito.passes.iet.openmp import ParallelRegion
from devito.tools import as_tuple
from devito.types import Scalar

pytestmark = skipif(['yask', 'ops'])


def get_blocksizes(op, dle, grid, blockshape, level=0):
    blocksizes = {'%s0_blk%d_size' % (d, level): v
                  for d, v in zip(grid.dimensions, blockshape)}
    blocksizes = {k: v for k, v in blocksizes.items() if k in op._known_arguments}
    # Sanity check
    if grid.dim == 1 or len(blockshape) == 0:
        assert len(blocksizes) == 0
        return {}
    try:
        if dle[1].get('blockinner'):
            assert len(blocksizes) >= 1
            if grid.dim == len(blockshape):
                assert len(blocksizes) == len(blockshape)
            else:
                assert len(blocksizes) <= len(blockshape)
        return blocksizes
    except AttributeError:
        assert len(blocksizes) == 0
        return {}


def _new_operator2(shape, time_order, blockshape=None, dle=None):
    blockshape = as_tuple(blockshape)
    grid = Grid(shape=shape, dtype=np.int32)
    infield = TimeFunction(name='infield', grid=grid, time_order=time_order)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = TimeFunction(name='outfield', grid=grid, time_order=time_order)

    stencil = Eq(outfield.forward.indexify(),
                 outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=dle)

    blocksizes = get_blocksizes(op, dle, grid, blockshape)
    op(infield=infield, outfield=outfield, t=10, **blocksizes)

    return outfield, op


def _new_operator3(shape, blockshape0=None, blockshape1=None, dle=None):
    blockshape0 = as_tuple(blockshape0)
    blockshape1 = as_tuple(blockshape1)

    grid = Grid(shape=shape, extent=shape, dtype=np.float64)

    # Allocate the grid and set initial condition
    # Note: This should be made simpler through the use of defaults
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=(2, 2, 2))
    u.data[0, :] = np.linspace(-1, 1, reduce(mul, shape)).reshape(shape)

    # Derive the stencil according to devito conventions
    op = Operator(Eq(u.forward, 0.5 * u.laplace + u), dle=dle)

    blocksizes0 = get_blocksizes(op, dle, grid, blockshape0, 0)
    blocksizes1 = get_blocksizes(op, dle, grid, blockshape1, 1)
    op.apply(u=u, t=10, **blocksizes0, **blocksizes1)

    return u.data[1, :], op


@pytest.mark.parametrize("shape", [(41,), (20, 33), (45, 31, 45)])
def test_composite_transformation(shape):
    wo_blocking, _ = _new_operator2(shape, time_order=2, dle='noop')
    w_blocking, _ = _new_operator2(shape, time_order=2, dle='advanced')

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("blockinner,exp_calls,exp_iters", [
    (False, 4, 5),
    (True, 8, 6)
])
@patch("devito.passes.iet.openmp.Ompizer.COLLAPSE_NCORES", 1)
def test_cache_blocking_structure(blockinner, exp_calls, exp_iters):
    # Check code structure
    _, op = _new_operator2((10, 31, 45), time_order=2,
                           dle=('blocking', {'blockinner': blockinner}))
    calls = FindNodes(Call).visit(op)
    assert len(calls) == exp_calls
    trees = retrieve_iteration_tree(op._func_table['bf0'].root)
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree) == exp_iters
    assert isinstance(tree[0].dim, BlockDimension)
    assert isinstance(tree[1].dim, BlockDimension)
    if blockinner:
        assert isinstance(tree[2].dim, BlockDimension)
    else:
        assert not isinstance(tree[2].dim, BlockDimension)
    assert not isinstance(tree[3].dim, BlockDimension)
    assert not isinstance(tree[4].dim, BlockDimension)

    # Check presence of openmp pragmas at the right place
    _, op = _new_operator2((10, 31, 45), time_order=2,
                           dle=('blocking', {'openmp': True, 'blockinner': blockinner}))
    trees = retrieve_iteration_tree(op._func_table['bf0'].root)
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree.root.pragmas) == 1
    assert 'omp for' in tree.root.pragmas[0].value
    # Also, with omp parallelism enabled, the step increment must be != 0
    # to avoid omp segfaults at scheduling time (only certain omp implementations,
    # including Intel's)
    conditionals = FindNodes(Conditional).visit(op._func_table['bf0'].root)
    assert len(conditionals) == 1
    conds = conditionals[0].condition.args
    expected_guarded = tree[:2+blockinner]
    assert len(conds) == len(expected_guarded)
    assert all(i.lhs == j.step for i, j in zip(conds, expected_guarded))


def test_cache_blocking_structure_subdims():
    """
    Test that:

        * With local SubDimensions no-blocking is expected.
        * With non-local SubDimensions, blocking is expected.
    """
    grid = Grid(shape=(4, 4, 4))
    x, y, z = grid.dimensions
    t = grid.stepping_dim
    xl = SubDimension.left(name='xl', parent=x, thickness=4)

    f = TimeFunction(name='f', grid=grid)

    assert xl.local

    # Local SubDimension -> no blocking expected
    op = Operator(Eq(f[t+1, xl, y, z], f[t, xl, y, z] + 1))
    assert len(op._func_table) == 0

    # Non-local SubDimension -> blocking expected
    op = Operator(Eq(f.forward, f + 1, subdomain=grid.interior))
    trees = retrieve_iteration_tree(op._func_table['bf0'].root)
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree) == 5
    assert isinstance(tree[0].dim, BlockDimension) and tree[0].dim.root is x
    assert isinstance(tree[1].dim, BlockDimension) and tree[1].dim.root is y
    assert not isinstance(tree[2].dim, BlockDimension)


@pytest.mark.parametrize("shape", [(10,), (10, 45), (20, 33), (10, 31, 45), (45, 31, 45)])
@pytest.mark.parametrize("time_order", [2])
@pytest.mark.parametrize("blockshape", [2, (3, 3), (9, 20), (2, 9, 11), (7, 15, 23)])
@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_time_loop(shape, time_order, blockshape, blockinner):
    wo_blocking, _ = _new_operator2(shape, time_order, dle='noop')
    w_blocking, _ = _new_operator2(shape, time_order, blockshape,
                                   dle=('blocking', {'blockinner': blockinner}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape,blockshape", [
    ((25, 25, 46), (25, 25, 46)),
    ((25, 25, 46), (7, 25, 46)),
    ((25, 25, 46), (25, 25, 7)),
    ((25, 25, 46), (25, 7, 46)),
    ((25, 25, 46), (5, 25, 7)),
    ((25, 25, 46), (10, 3, 46)),
    ((25, 25, 46), (25, 7, 11)),
    ((25, 25, 46), (8, 2, 4)),
    ((25, 25, 46), (2, 4, 8)),
    ((25, 25, 46), (4, 8, 2)),
    ((25, 46), (25, 7)),
    ((25, 46), (7, 46))
])
def test_cache_blocking_edge_cases(shape, blockshape):
    time_order = 2
    wo_blocking, _ = _new_operator2(shape, time_order, dle='noop')
    w_blocking, _ = _new_operator2(shape, time_order, blockshape,
                                   dle=('blocking', {'blockinner': True}))
    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape,blockshape", [
    ((3, 3), (3, 3)),
    ((4, 4), (3, 4)),
    ((5, 5), (3, 4)),
    ((6, 6), (3, 4)),
    ((7, 7), (3, 4)),
    ((8, 8), (3, 4)),
    ((9, 9), (3, 4)),
    ((10, 10), (3, 4)),
    ((11, 11), (3, 4)),
    ((12, 12), (3, 4)),
    ((13, 13), (3, 4)),
    ((14, 14), (3, 4)),
    ((15, 15), (3, 4))
])
def test_cache_blocking_edge_cases_highorder(shape, blockshape):
    wo_blocking, a = _new_operator3(shape, dle='noop')
    w_blocking, b = _new_operator3(shape, blockshape, dle=('blocking',
                                                           {'blockinner': True}))

    assert np.allclose(wo_blocking, w_blocking, rtol=1e-12)


@pytest.mark.parametrize("blockshape0,blockshape1,exception", [
    ((24, 24, 40), (24, 24, 40), False),
    ((24, 24, 40), (4, 4, 4), False),
    ((24, 24, 40), (8, 8, 8), False),
    ((20, 20, 12), (4, 4, 4), False),
    ((28, 32, 16), (14, 16, 8), False),
    ((12, 12, 60), (4, 12, 4), False),
    ((12, 12, 60), (4, 5, 4), True),  # not a perfect divisor
    ((12, 12, 60), (24, 4, 4), True),  # bigger than outer block
])
def test_cache_blocking_hierarchical(blockshape0, blockshape1, exception):
    shape = (51, 102, 71)

    wo_blocking, a = _new_operator3(shape, dle='noop')
    try:
        w_blocking, b = _new_operator3(shape, blockshape0, blockshape1,
                                       dle=('blocking', {'blockinner': True,
                                                         'blocklevels': 2}))
        assert not exception
        assert np.allclose(wo_blocking, w_blocking, rtol=1e-12)
    except InvalidArgument:
        assert exception
    except:
        assert False


def test_cache_blocking_unsupported():
    """
    Test that a non-perfect Iteration nest is not blocked.
    """
    grid = Grid(shape=(4, 4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)
    v = TimeFunction(name='v', grid=grid, space_order=2)

    eqns = [Eq(u.forward, v.laplace),
            Eq(v.forward, u.forward.dz)]

    op = Operator(eqns)

    assert not op._func_table

    trees = retrieve_iteration_tree(op)
    assert len(trees) == 2


class TestNodeParallelism(object):

    @pytest.mark.parametrize('exprs,expected', [
        # trivial 1D
        (['Eq(fa[x], fa[x] + fb[x])'],
         (True,)),
        # trivial 1D
        (['Eq(t0, fa[x] + fb[x])', 'Eq(fa[x], t0 + 1)'],
         (True,)),
        # trivial 2D
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x,y], t0 + 1)'],
         (True, False)),
        # outermost parallel, innermost sequential
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x,y+1], t0 + 1)'],
         (True, False)),
        # outermost sequential, innermost parallel
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x+1,y], t0 + 1)'],
         (False, True)),
        # outermost sequential, innermost parallel
        (['Eq(fc[x,y], fc[x+1,y+1] + fc[x-1,y])'],
         (False, True)),
        # outermost parallel w/ repeated dimensions, but the compiler is conservative
        # and makes it sequential, as it doesn't like what happens in the inner dims,
        # where `x`, rather than `y`, is used. The innermost one is instead parallel,
        # as there are no deps along `y`.
        (['Eq(t0, fc[x,x] + fd[x,y+1])', 'Eq(fc[x,x], t0 + 1)'],
         (False, True)),
        # outermost sequential, innermost sequential (classic skewing example)
        (['Eq(fc[x,y], fc[x,y+1] + fc[x-1,y])'],
         (False, False)),
        # skewing-like over two Eqs
        (['Eq(t0, fc[x,y+2] + fc[x-1,y+2])', 'Eq(fc[x,y+1], t0 + 1)'],
         (False, False)),
        # outermost parallel, innermost sequential w/ double tensor write
        (['Eq(fc[x,y], fc[x,y+1] + fd[x-1,y])', 'Eq(fd[x-1,y+1], fd[x-1,y] + fc[x,y+1])'],
         (True, False, False)),
        # outermost sequential, innermost parallel w/ mixed dimensions
        (['Eq(fc[x+1,y], fc[x,y+1] + fc[x,y])', 'Eq(fc[x+1,y], 2. + fc[x,y+1])'],
         (False, True)),
    ])
    def test_iterations_ompized(self, exprs, expected):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions  # noqa

        fa = Function(name='fa', grid=grid, dimensions=(x,), shape=(4,))  # noqa
        fb = Function(name='fb', grid=grid, dimensions=(x,), shape=(4,))  # noqa
        fc = Function(name='fc', grid=grid)  # noqa
        fd = Function(name='fd', grid=grid)  # noqa
        t0 = Scalar(name='t0')  # noqa

        eqns = []
        for e in exprs:
            eqns.append(eval(e))

        op = Operator(eqns, dle='openmp')

        iterations = FindNodes(Iteration).visit(op)
        assert len(iterations) == len(expected)

        # Check for presence of pragma omp
        for i, j in zip(iterations, expected):
            pragmas = i.pragmas
            if j is True:
                assert len(pragmas) == 1
                pragma = pragmas[0]
                assert 'omp for' in pragma.value
            else:
                for k in pragmas:
                    assert 'omp for' not in k.value

    def test_dynamic_nthreads(self):
        grid = Grid(shape=(16, 16, 16))
        f = TimeFunction(name='f', grid=grid)
        sf = SparseTimeFunction(name='sf', grid=grid, npoint=1, nt=5)

        eqns = [Eq(f.forward, f + 1)]
        eqns += sf.interpolate(f)

        op = Operator(eqns, dle='openmp')

        parregions = FindNodes(ParallelRegion).visit(op)
        assert len(parregions) == 2

        # Check suitable `num_threads` appear in the generated code
        # Not very elegant, but it does the trick
        assert 'num_threads(nthreads)' in str(parregions[0].header[0])
        assert 'num_threads(nthreads_nonaffine)' in str(parregions[1].header[0])

        # Check `op` accepts the `nthreads*` kwargs
        op.apply(time=0)
        op.apply(time_m=1, time_M=1, nthreads=4)
        op.apply(time_m=1, time_M=1, nthreads=4, nthreads_nonaffine=2)
        op.apply(time_m=1, time_M=1, nthreads_nonaffine=2)
        assert np.all(f.data[0] == 2.)

        # Check the actual value assumed by `nthreads` and `nthreads_nonaffine`
        assert op.arguments(time=0)['nthreads'] == NThreads.default_value()
        assert op.arguments(time=0)['nthreads_nonaffine'] == \
            NThreadsNonaffine.default_value()
        # Again, but with user-supplied values
        assert op.arguments(time=0, nthreads=123)['nthreads'] == 123
        assert op.arguments(time=0, nthreads_nonaffine=100)['nthreads_nonaffine'] == 100
        # Again, but with the aliases
        assert op.arguments(time=0, nthreads0=123)['nthreads'] == 123
        assert op.arguments(time=0, nthreads2=123)['nthreads_nonaffine'] == 123

    @pytest.mark.parametrize('eqns,expected,blocking', [
        ('[Eq(f, 2*f)]', [2, 0, 0], False),
        ('[Eq(u, 2*u)]', [0, 2, 0, 0], False),
        ('[Eq(u, 2*u)]', [3, 0, 0, 0, 0, 0], True),
        ('[Eq(u, 2*u), Eq(f, u.dzr)]', [0, 2, 0, 0, 0], False)
    ])
    @patch("devito.passes.iet.openmp.Ompizer.COLLAPSE_NCORES", 1)
    @patch("devito.passes.iet.openmp.Ompizer.COLLAPSE_WORK", 0)
    def test_collapsing(self, eqns, expected, blocking):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)  # noqa
        u = TimeFunction(name='u', grid=grid)  # noqa

        eqns = eval(eqns)

        if blocking:
            op = Operator(eqns, dle=('blocking', 'simd', 'openmp', {'blockinner': True}))
            iterations = FindNodes(Iteration).visit(op._func_table['bf0'])
        else:
            op = Operator(eqns, dle=('simd', 'openmp'))
            iterations = FindNodes(Iteration).visit(op)

        assert len(iterations) == len(expected)

        # Check for presence of pragma omp + collapse clause
        for i, j in zip(iterations, expected):
            if j > 0:
                assert len(i.pragmas) == 1
                pragma = i.pragmas[0]
                assert 'omp for collapse(%d)' % j in pragma.value
            else:
                for k in i.pragmas:
                    assert 'omp for collapse' not in k.value

    def test_scheduling(self):
        """
        Affine iterations -> #pragma omp ... schedule(dynamic,1) ...
        Non-affine iterations -> #pragma omp ... schedule(dynamic,chunk_size) ...
        """
        grid = Grid(shape=(11, 11))

        u = TimeFunction(name='u', grid=grid, time_order=2, save=5, space_order=0)
        sf1 = SparseTimeFunction(name='s', grid=grid, npoint=1, nt=5)

        eqns = [Eq(u.forward, u + 1)]
        eqns += sf1.interpolate(u)

        op = Operator(eqns, dle='openmp')

        iterations = FindNodes(Iteration).visit(op)
        assert len(iterations) == 4
        assert iterations[1].is_Affine
        assert 'schedule(dynamic,1)' in iterations[1].pragmas[0].value
        assert not iterations[3].is_Affine
        assert 'schedule(dynamic,chunk_size)' in iterations[3].pragmas[0].value


class TestNestedParallelism(object):

    @patch("devito.passes.iet.openmp.Ompizer.NESTED", 0)
    @patch("devito.passes.iet.openmp.Ompizer.COLLAPSE_NCORES", 10000)
    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), dle=('blocking', 'openmp'))

        # Does it compile? Honoring the OpenMP specification isn't trivial
        assert op.cfunction

        # Does it produce the right result
        op.apply(t_M=9)
        assert np.all(u.data[0] == 10)

        # Try again but this time supplying specific values for the num_threads
        u.data[:] = 0.
        op.apply(t_M=9, nthreads=1, nthreads_nested=2)
        assert np.all(u.data[0] == 10)
        assert op.arguments(t_M=9, nthreads_nested=2)['nthreads_nested'] == 2
        # Same as above, but with the alias
        assert op.arguments(t_M=9, nthreads1=2)['nthreads_nested'] == 2

        iterations = FindNodes(Iteration).visit(op._func_table['bf0'])
        assert iterations[0].pragmas[0].value == 'omp for collapse(1) schedule(dynamic,1)'
        assert iterations[2].pragmas[0].value == ('omp parallel for collapse(1) '
                                                  'schedule(dynamic,1) '
                                                  'num_threads(nthreads_nested)')

    @patch("devito.passes.iet.openmp.Ompizer.NESTED", 0)
    @patch("devito.passes.iet.openmp.Ompizer.COLLAPSE_NCORES", 1)
    @patch("devito.passes.iet.openmp.Ompizer.COLLAPSE_WORK", 0)
    def test_collapsing(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), dle=('blocking', 'openmp'))

        # Does it compile? Honoring the OpenMP specification isn't trivial
        assert op.cfunction

        # Does it produce the right result
        op.apply(t_M=9)
        assert np.all(u.data[0] == 10)

        iterations = FindNodes(Iteration).visit(op._func_table['bf0'])
        assert iterations[0].pragmas[0].value == 'omp for collapse(2) schedule(dynamic,1)'
        assert iterations[2].pragmas[0].value == ('omp parallel for collapse(2) '
                                                  'schedule(dynamic,1) '
                                                  'num_threads(nthreads_nested)')

    @patch("devito.passes.clusters.aliases.MIN_COST_ALIAS", 1)
    @patch("devito.passes.iet.openmp.Ompizer.NESTED", 0)
    @patch("devito.passes.iet.openmp.Ompizer.COLLAPSE_NCORES", 10000)
    def test_multiple_subnests(self):
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid)

        eqn = Eq(u.forward, ((u[t, x, y, z] + u[t, x+1, y+1, z+1])*3*f +
                             (u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3*f + 1))
        op = Operator(eqn, dse='aggressive', dle=('advanced', {'openmp': True}))

        trees = retrieve_iteration_tree(op._func_table['bf0'].root)
        assert len(trees) == 2

        assert trees[0][0] is trees[1][0]
        assert trees[0][0].pragmas[0].value ==\
            'omp for collapse(1) schedule(dynamic,1)'
        assert trees[0][2].pragmas[0].value == ('omp parallel for collapse(1) '
                                                'schedule(dynamic,1) '
                                                'num_threads(nthreads_nested)')
        assert trees[1][2].pragmas[0].value == ('omp parallel for collapse(1) '
                                                'schedule(dynamic,1) '
                                                'num_threads(nthreads_nested)')


class TestOffloading(object):

    """
    Test GPU offloading via OpenMP.
    """

    @switchconfig(platform='nvidiaX')
    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), dle=('advanced', {'openmp': True}))

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        assert op.body[1].header[0].value ==\
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
            assert op.body[2].header[i].value ==\
                ('omp target enter data map(to: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})
            assert op.body[2].footer[i].value ==\
                ('omp target exit data map(from: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})

    @switchconfig(platform='nvidiaX')
    def test_multiple_loops(self):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=2)
        v = TimeFunction(name='v', grid=grid, space_order=2)

        eqns = [Eq(f, g*2),
                Eq(u.forward, u + v*f),
                Eq(v.forward, u.forward.dx + v*f + 4)]

        op = Operator(eqns, dle=('noop', {'openmp': True}))

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 3

        # All loop nests must have been parallelized
        assert trees[0][0].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        assert trees[1][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        assert trees[2][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'

        # Check `u` and `v`
        for i, f in enumerate([u, v], 1):
            assert op.body[4].header[i].value ==\
                ('omp target enter data map(to: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})
            assert op.body[4].footer[i].value ==\
                ('omp target exit data map(from: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})

        # Check `f`
        assert op.body[4].header[0].value ==\
            ('omp target enter data map(to: f[0:f_vec->size[0]]'
             '[0:f_vec->size[1]][0:f_vec->size[2]])')
        assert op.body[4].footer[0].value ==\
            ('omp target exit data map(from: f[0:f_vec->size[0]]'
             '[0:f_vec->size[1]][0:f_vec->size[2]])')

        # Check `g` -- note that unlike `f`, this one should be `delete` upon
        # exit, not `from`
        assert op.body[4].header[3].value ==\
            ('omp target enter data map(to: g[0:g_vec->size[0]]'
             '[0:g_vec->size[1]][0:g_vec->size[2]])')
        assert op.body[4].footer[3].value ==\
            ('omp target exit data map(delete: g[0:g_vec->size[0]]'
             '[0:g_vec->size[1]][0:g_vec->size[2]])')

    @switchconfig(platform='nvidiaX')
    def test_array_rw(self):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=2)

        eqn = Eq(u.forward, u*cos(f*2))

        op = Operator(eqn, dse='aggressive', dle=('noop', {'openmp': True}))

        assert len(op.body[2].header) == 4
        assert str(op.body[2].header[0]) == 'float (*r1)[y_size][z_size];'
        assert op.body[2].header[1].contents[0].text ==\
            'posix_memalign((void**)&r1, 64, sizeof(float[x_size][y_size][z_size]))'
        assert op.body[2].header[1].contents[1].value ==\
            'omp target enter data map(alloc: r1[0:x_size][0:y_size][0:z_size])'

        assert len(op.body[2].footer) == 3
        assert op.body[2].footer[0].contents[0].value ==\
            'omp target exit data map(delete: r1[0:x_size][0:y_size][0:z_size])'
        assert op.body[2].footer[0].contents[1].text == 'free(r1)'

    @switchconfig(platform='nvidiaX')
    def test_function_wo(self):
        grid = Grid(shape=(3, 3, 3))
        i = Dimension(name='i')

        f = Function(name='f', shape=(1,), dimensions=(i,), grid=grid)
        u = TimeFunction(name='u', grid=grid)

        eqns = [Eq(u.forward, u + 1),
                Eq(f[0], u[0, 0, 0, 0])]

        op = Operator(eqns, dle=('noop', {'openmp': True}))

        assert len(op.body[2].header) == 1
        assert len(op.body[2].footer) == 1
        assert op.body[2].header[0].value ==\
            ('omp target enter data map(to: u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body[2].footer[0].value ==\
            ('omp target exit data map(from: u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')

    @switchconfig(platform='nvidiaX')
    def test_timeparallel_reduction(self):
        grid = Grid(shape=(3, 3, 3))
        i = Dimension(name='i')

        f = Function(name='f', shape=(1,), dimensions=(i,), grid=grid)
        u = TimeFunction(name='u', grid=grid)

        op = Operator(Inc(f[0], u + 1), dle=('noop', {'openmp': True}))

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1
        tree = trees[0]
        assert all(i.is_ParallelRelaxed and not i.is_Parallel for i in tree)

        # The time loop is not in OpenMP canonical form, so it won't be parallelized
        assert not tree.root.pragmas
        assert len(tree[1].pragmas) == 1
        assert tree[1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3) reduction(+:f[0])'
