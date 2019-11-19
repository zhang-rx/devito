from devito.symbolics.search import retrieve_indexed, search
from sympy import diff
from devito.types import TimeFunction, Dimension
from devito.ir.equations.equation import LoweredEq
from devito.ir.support.space import (DataSpace, Interval, IntervalGroup,
                                     IterationSpace, AbstractInterval, NullInterval)
from devito.types.dimension import SubDimension

def differentiate(expressions):
    all_derivatives = []
    for e in expressions:

        derivatives = differentiate_expression(e)

        derivatives = make_same_lhs(derivatives, e)

        dim_index = 1
        intervals = [d.ispace.intervals[dim_index] for d in derivatives]

        intersection = IntervalGroup.generate('intersection', *[IntervalGroup(x) for x in intervals])
        assert(not intersection[0].is_Null)
        d = derivatives[0]
        dim = retrieve_dimension(d.lhs.indices[dim_index]).pop()

        new_equations = []
        for d in derivatives:
            left_interval = left_remainder(d.ispace.intervals[dim_index], intersection[0])
            if not left_interval.is_Null:
                left_eq = replace_dimension_left(d, dim, left_interval)
                new_equations.append(left_eq)
                
            middle_eq = replace_dimension_middle(d, dim, intersection[0])
            new_equations.append(middle_eq)
            
            right_interval = right_remainder(d.ispace.intervals[dim_index], intersection[0])
            if not right_interval.is_Null:
                right_eq = replace_dimension_right(d, dim, right_interval)
                new_equations.append(right_eq)

        derivatives = new_equations

        all_derivatives += derivatives
    return all_derivatives

def replace_dimension_left(expression, dim, interval):
    return replace_dimension(expression, dim, interval, 'left')
    
def replace_dimension_middle(expression, dim, interval):
    return replace_dimension(expression, dim, interval, 'middle')
    
def replace_dimension_right(expression, dim, interval):
    return replace_dimension(expression, dim, interval, 'right')

def replace_dimension(expression, dim, interval, direction):
    assert(direction in ['left', 'right', 'middle'])
    thickness0 = interval.lower
    thickness1 = interval.upper
    if direction=='left':
        new_dim = SubDimension.leftleft(parent=dim, name="%s_left"%dim.name, thickness0=thickness0, thickness1=thickness1)
    elif direction=='middle':
        new_dim = SubDimension.middle(parent=dim, name="%s_middle"%dim.name, thickness_left=thickness0, thickness_right=thickness1)
    else:
        new_dim = SubDimension.rightright(parent=dim, name="%s_right"%dim.name, thickness0=thickness0, thickness1=thickness1)
    return expression.subs({dim:new_dim})

def left_remainder(i1, i2):
    assert(i1.dim == i2.dim)
    if not i1.lower < i2.lower:
        return NullInterval(i1.dim)
    
    return Interval(i1.dim, i1.lower, i2.lower)


def differentiate_expression(e):
    adjoint_mapper = {}
    indexeds = retrieve_indexed(e.rhs, mode='all', deep=True)
    adjoint_output_fn, adjoint_mapper = diff_indexed(e.lhs, adjoint_mapper)
        
    derivatives = []
    for i in indexeds:
        i_d, adjoint_mapper = diff_indexed(i, adjoint_mapper)
        state = extract_le_state(e)

        old_ds = state['dspace']

        new_parts = {i_d.function: old_ds.parts[i.function]}
        new_ds = DataSpace(old_ds.intervals, new_parts)
        state['dspace'] = new_ds

        d_eqn = LoweredEq(i_d, i_d+diff(e.rhs, i)*adjoint_output_fn, **state)
        derivatives.append(d_eqn)
    return derivatives

def make_same_lhs(derivatives, original_e):
    new_derivatives = []
    for i, d in enumerate(derivatives):
        subs = {}
        for e_ind, d_ind in zip(original_e.lhs.indices, d.lhs.indices):
            ind = retrieve_dimension(d_ind).pop()
            if ind.is_Time:
                continue
            else:
                if (d_ind - e_ind) != 0:
                    subs[ind] = e_ind - d_ind
        new_derivatives.append(shift_le_index(d, subs))
    ensure_single_lhs(new_derivatives)
    return new_derivatives

def right_remainder(i1, i2):
    assert(i1.dim == i2.dim)
    if not i1.upper > i2.upper:
        return NullInterval(i1.dim)

    return Interval(i1.dim, i2.upper, i1.upper)

def ensure_single_lhs(derivatives):
    lhs_terms = set()

    for d in derivatives:
        lhs_terms.add(d.lhs)

    assert(len(lhs_terms) == 1)
    
def diff_function(func, existing_mapper):
    if func in existing_mapper:
        return existing_mapper[func], existing_mapper
    else:
        adjoint_fn = TimeFunction(name="%sd" % func.name, space_order=func.space_order,
                                  time_order=func.time_order, grid=func.grid)
        existing_mapper[func] = adjoint_fn
        return adjoint_fn, existing_mapper


def diff_indexed(indexed, fn_mapper):
    adjoint_fn, fn_mapper = diff_function(indexed.function, fn_mapper)
    return adjoint_fn[tuple(indexed.indices)], fn_mapper


def extract_le_state(lowered_eq):
    state = {}
    for i in lowered_eq._state:
        state[i] = getattr(lowered_eq, i)
    return state


def shift_le_index(le, mapper):
    s_m = dict([(k, k+v) for k, v in mapper.items()])
    state = extract_le_state(le)
    ds = state['dspace']
    new_parts = {le.lhs.function: shift_interval_group(ds.parts[le.lhs.function], mapper)}
    new_ds = DataSpace(shift_interval_group(ds.intervals, mapper), new_parts)

    ispace = state['ispace']
    new_ispace = IterationSpace(shift_interval_group(ispace.intervals, mapper),
                                ispace.sub_iterators, ispace.directions)

    state['dspace'] = new_ds
    state['ispace'] = new_ispace

    new_le = LoweredEq(le.lhs.subs(s_m), le.rhs.subs(s_m), **state)
    return new_le


def shift_interval_group(interval_group, mapper):
    new_intervals = []
    for interval in interval_group:
        if interval.dim in mapper:
            new_intervals.append(Interval(interval.dim,
                                          interval.lower+mapper[interval.dim],
                                          interval.upper+mapper[interval.dim]))
        else:
            new_intervals.append(interval)
    return IntervalGroup(new_intervals)


def q_dimension(expr):
    return isinstance(expr, Dimension)


def retrieve_dimension(expr, mode='unique', deep=False):
    """Shorthand to retrieve the Dimensions in ``expr``."""
    return search(expr, q_dimension, mode, 'dfs', deep)
