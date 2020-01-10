from functools import reduce
from operator import mul

from sympy import Add
import numpy as np
import pytest
from unittest.mock import patch

from conftest import skipif
from devito import (Grid, Function, TimeFunction, SparseTimeFunction, SubDimension,
                    Eq, Operator, switchconfig)
from devito.exceptions import InvalidArgument
from devito.ir.iet import (Call, Iteration, Conditional, FindNodes, FindSymbols,
                           retrieve_iteration_tree)
from devito.targets import BlockDimension, NThreads, NThreadsNonaffine
from devito.targets.common.openmp import ParallelRegion
from devito.tools import as_tuple
from devito.types import Scalar

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

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), dle=('advanced', {'openmp': True}))

        time_steps = 4
        op.apply(time_M=time_steps)

        assert np.all(u.data[:] == time_steps)
