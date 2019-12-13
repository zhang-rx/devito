import numpy as np

from devito import TimeFunction, Grid, Eq, Function, Dimension, SubDimension
from devito.data.allocators import ExternalAllocator
from devito.ir.equations import LoweredEq
from devito.ir.iet import Expression, FindNodes, List
from devito.ir.iet.visitors import Transformer
from devito.operator import Operator
from devito.symbolics import retrieve_indexed

from devito.types import Symbol

from utils import generate_data, external_initializer


class InPlaceOperator(Operator):

    def __init__(self, *args, **kwargs):
        super(InPlaceOperator, self).__init__(*args, **kwargs)

    def _specialize_exprs(self, expressions):

        # building symbols to hold input data
        in1 = Symbol(name='in1')
        in2 = Symbol(name='in2')

        # building functions to hold output data
        expr_in = expressions[0]
        grid_in = expr_in.lhs.function.grid
        out1 = Function(name='out1', grid=grid_in)
        out2 = Function(name='out2', grid=grid_in)

        # extract symbols from input expression
        symb = retrieve_indexed(expr_in)
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
        exprs_out = []
        tmp_expr = expr_in
        for i in range(5):        
            tmp_expr = tmp_expr.xreplace(mapping[i])
            exprs_out.append(tmp_expr)


        # building list of subdimensions
        t = expr_in.lhs.function.time_dim
        t_length = t.extreme_max - t.extreme_min
        t0 = SubDimension.left(name='t1', parent=t, thickness=1)
        t1 = SubDimension.middle(name='t2', parent=t, thickness_left=1, thickness_right=t_length-2)
        t2 = SubDimension.middle(name='t3', parent=t, thickness_left=2, thickness_right=2)
        t3 = SubDimension.middle(name='t4', parent=t, thickness_left=t_length-2, thickness_right=1)
        t4 = SubDimension.right(name='t5', parent=t, thickness=1)
        
        # applying subdimensions
        exprs_out[0] = exprs_out[0].subs(t,t0)
        exprs_out[1] = exprs_out[1].subs(t,t1)
        exprs_out[2] = exprs_out[2].subs(t,t2) # this one will cause fail!
        # exprs_out[3] = exprs_out[3].subs(t,t3)
        # exprs_out[4] = exprs_out[4].subs(t,t4)

        # debuggin
        for i in range(5):        
            print("%s) %s" % (i,exprs_out[i]))

        return [LoweredEq(i) for i in exprs_out]
        

## Interface

shape=(10,10)
space_order=2
time_order=2
save=2
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

# print(operator)