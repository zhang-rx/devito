import numpy as np

from devito import TimeFunction, Grid, Eq, Function, Dimension, SubDimension
from devito.data.allocators import ExternalAllocator
from devito.ir.equations import LoweredEq
from devito.ir.iet import Expression, FindNodes, List
from devito.ir.iet.visitors import Transformer
from devito.operator import Operator
from devito.symbolics import retrieve_indexed, indexify
from devito.tools import flatten
from devito.types import Symbol

from utils import generate_data, external_initializer


def _func_args(equation):
    """
    Retrieves all functions from a given equation.
    """
    if equation.is_Function:
        return equation
    elif equation.is_Equality:
        return [_func_args(equation.lhs),
                _func_args(equation.rhs)]
    elif equation.is_Add:
        return [_func_args(i) for i in equation.args if not(i.is_Number)]


class InPlaceOperator(Operator):

    def __init__(self, *args, **kwargs):

        eq_in = args[0]                     # input equation
        symb = flatten(_func_args(args[0])) # retrieving all functions
        grid_in = symb[0].grid              # grid info

        # building functions to hold input/output data
        in1 = Function(name='in1', grid=grid_in)
        in2 = Function(name='in2', grid=grid_in)
        out1 = Function(name='out1', grid=grid_in)
        out2 = Function(name='out2', grid=grid_in)

        symb_ordered = [symb[1], symb[2], symb[0]] # TODO: a function to order symb 
        symb_ordered_ext = [in2, in1] + symb_ordered + [out1, out2]

        # drawing successive symbol shifts
        # first pass
        mapping = []
        mapping.append({symb_ordered_ext[2]:symb_ordered_ext[0],
                        symb_ordered_ext[3]:symb_ordered_ext[1],
                        symb_ordered_ext[4]:symb_ordered_ext[2]})
        # remaining passes
        for i in range(4):
            mapping.append({symb_ordered_ext[i+2]:symb_ordered_ext[i+3],
                            symb_ordered_ext[i+1]:symb_ordered_ext[i+2],
                            symb_ordered_ext[i+0]:symb_ordered_ext[i+1]})
        
        # building list of output expressions
        eqs_out = []
        tmp_eq = eq_in
        for i in range(5):        
            tmp_eq = tmp_eq.xreplace(mapping[i])
            eqs_out.append(tmp_eq)

        # building list of subdimensions
        t = eq_in.lhs.time_dim
        t_length = t.extreme_max - t.extreme_min
        thickness0 = 1
        thickness1 = 2
        # from IPython import embed; embed()
        t0 = SubDimension.left(name='t0',
                               parent=t,
                               thickness=thickness0)
        t1 = SubDimension.middle(name='t1',
                               parent=t,
                               thickness_left=thickness0,
                               thickness_right=t_length-thickness1)
        t2 = SubDimension.middle(name='t2',
                               parent=t,
                               thickness_left=thickness1,
                               thickness_right=thickness1)
        t3 = SubDimension.middle(name='t3',
                               parent=t,
                               thickness_left=t_length-thickness1,
                               thickness_right=thickness0)
        t4 = SubDimension.right(name='t4',
                               parent=t,
                               thickness=thickness0)

        # applying subdimensions
        eqs_out[0] = eqs_out[0].subs(t,t0)
        eqs_out[1] = eqs_out[1].subs(t,t1)
        eqs_out[2] = eqs_out[2].subs(t,t2)
        eqs_out[3] = eqs_out[3].subs(t,t3)
        eqs_out[4] = eqs_out[4].subs(t,t4)

        super(InPlaceOperator, self).__init__(tuple(eqs_out), **kwargs)


## Interface

shape=(10,10)
space_order=0
time_order=2
save=1
numpy_array, devito_grid = generate_data(shape=shape, 
                                         space_order=space_order, 
                                         save=save)

f1 = TimeFunction(name='f1',
                  grid=devito_grid,
                  space_order=space_order,
                  allocator=ExternalAllocator(numpy_array),
                  initializer=external_initializer,
                  time_order=time_order,
                  save=save)

equation = Eq(f1.forward, f1 + f1.backward + 1)
operator = InPlaceOperator(equation, time_m=0, time_M=9)

print(operator)