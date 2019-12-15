from devito import Eq, Grid, TimeFunction  # noqa
from devito.data.allocators import ExternalAllocator
from devito.ir.iet import retrieve_iteration_tree

from studies.inplace import InPlaceOperator
from studies.utils import external_initializer

import numpy as np
import pytest


@pytest.mark.parametrize('equation, expected', [
    ('Eq(f1.forward, f1 + f1.backward + 1)',
     '[\'f1[t0 - 1][x][y] = in1[x][y] + in2[x][y] + 1;\','
     '\'f1[t1][x][y] = in1[x][y] + f1[t1 - 1][x][y] + 1;\','
     '\'f1[t2 + 1][x][y] = f1[t2][x][y] + f1[t2 - 1][x][y] + 1;\','
     '\'out1[x][y] = f1[t3][x][y] + f1[t3 + 1][x][y] + 1;\','
     '\'out2[x][y] = out1[x][y] + f1[t4 + 1][x][y] + 1;\']')
])
def test_inplace(equation, expected):

    shape = (10, 10)
    grid = Grid(shape=shape)
    numpy_array = np.zeros(shape=shape, dtype=np.float32)

    space_order = 0  # current InPlaceOperator implementation is restricted to 0
    time_order = 2   # current InPlaceOperator implementation is reqqstricted to 2
    f1 = TimeFunction(name='f1',  # noqa 
                      grid=grid,
                      space_order=space_order,
                      allocator=ExternalAllocator(numpy_array),
                      initializer=external_initializer,
                      time_order=time_order,
                      save=None)

    eq = eval(equation)
    op = InPlaceOperator(eq)

    assert (len(retrieve_iteration_tree(op)) == len(eval(expected)))

    for i in eval(expected):
        assert i in str(op)
