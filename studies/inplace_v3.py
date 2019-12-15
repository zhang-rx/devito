from devito import TimeFunction, Eq, SubDimension, Function
from devito.data.allocators import ExternalAllocator
from devito.operator import Operator
from devito.tools import flatten
from utils import generate_data, external_initializer


def _func_args(equation):
    """
    Retrieve all Functions from a given equation.
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

        eq_in = args[0]                      # Input Equation
        symb = flatten(_func_args(args[0]))  # Retrieving all Functions
        grid_in = symb[0].grid               # Grid info

        # Building functions to hold input/output data
        in1 = Function(name='in1', grid=grid_in)
        in2 = Function(name='in2', grid=grid_in)
        out1 = Function(name='out1', grid=grid_in)
        out2 = Function(name='out2', grid=grid_in)

        symb_ordered = [symb[1], symb[2], symb[0]]  # TODO: a function to order symbs
        symb_ordered_ext = [in2, in1] + symb_ordered + [out1, out2]

        # Drawing successive symbol shifts
        # First pass
        mapping = []
        mapping.append({symb_ordered_ext[2]: symb_ordered_ext[0],
                        symb_ordered_ext[3]: symb_ordered_ext[1],
                        symb_ordered_ext[4]: symb_ordered_ext[2]})
        # Remaining passes
        for i in range(4):
            mapping.append({symb_ordered_ext[i+2]: symb_ordered_ext[i+3],
                            symb_ordered_ext[i+1]: symb_ordered_ext[i+2],
                            symb_ordered_ext[i+0]: symb_ordered_ext[i+1]})

        # Building list of output expressions
        eqs_out = []
        tmp_eq = eq_in
        for i in range(5):
            tmp_eq = tmp_eq.xreplace(mapping[i])
            eqs_out.append(tmp_eq)

        # Building list of subdimensions
        t = eq_in.lhs.time_dim
        t_length = t.extreme_max - t.extreme_min
        thickness0 = 1
        thickness1 = 2

        ti = []
        ti.append(SubDimension.left(name='t0',
                                    parent=t,
                                    thickness=thickness0))
        ti.append(SubDimension.middle(name='t1',
                                      parent=t,
                                      thickness_left=thickness0,
                                      thickness_right=t_length-thickness1))
        ti.append(SubDimension.middle(name='t2',
                                      parent=t,
                                      thickness_left=thickness1,
                                      thickness_right=thickness1))
        ti.append(SubDimension.middle(name='t3',
                                      parent=t,
                                      thickness_left=t_length-thickness1,
                                      thickness_right=thickness0))
        ti.append(SubDimension.right(name='t4',
                                     parent=t,
                                     thickness=thickness0))

        # Applying subdimensions
        for i in range(5):
            eqs_out[i] = eqs_out[i].subs(t, ti[i])

        super(InPlaceOperator, self).__init__(tuple(eqs_out), **kwargs)


# Interface

shape = (10, 10)
space_order = 0
time_order = 2
save = 1
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
