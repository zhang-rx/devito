from devito.symbolics.search import retrieve_indexed, search
from sympy import diff
from devito.types import TimeFunction, Dimension
from devito.ir.equations.equation import LoweredEq
from devito.ir.support.space import (DataSpace, Interval, IntervalGroup,
                                     IterationSpace, AbstractInterval)
from devito.types.dimension import SubDimension

def differentiate(expressions):
    adjoint_mapper = {}
    all_derivatives = []
    for e in expressions:
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

        for i, d in enumerate(derivatives):
            subs = {}
            for e_ind, d_ind in zip(e.lhs.indices, d.lhs.indices):
                ind = retrieve_dimension(d_ind).pop()
                if ind.is_Time:
                    continue
                else:
                    if (d_ind - e_ind) != 0:
                        subs[ind] = e_ind - d_ind
            derivatives[i] = shift_le_index(d, subs)

        lhs_terms = set()

        for d in derivatives:
            lhs_terms.add(d.lhs)

        assert(len(lhs_terms) == 1)

        lhs_term = lhs_terms.pop()

        dim_index = 1
        intervals = []
        for d in derivatives:
            intervals.append(d.ispace.intervals[dim_index])

        intersection = IntervalGroup.generate('intersection', *[IntervalGroup(x) for x in intervals])

        new_equations = [replace_interval(x, intersection[0]) for x in derivatives]

        # r[i] = u[i-2]+u[i-1]+u[i]+u[i+1]

        left_extent = abs(min([i.lower for i in intervals])) + 1
        right_extent = abs(max([i.upper for i in intervals])) + 1
        dim = retrieve_dimension(d.lhs.indices[dim_index]).pop()
        left_remainder = SubDimension.left("%s_lr" % dim.name, dim, left_extent)
        right_remainder = SubDimension.right("%s_rr" % dim.name, dim, right_extent)

        # i = Interval(xl, -1, 1)

        
        
        for d in derivatives:
            pass
            # left remainder interval
            # remainder_interval = left_remainder(d.ispace.intervals[dim_index], intersection[0])
            # left remainder equation
            # new_equation = 
            # new_equations.append(replace_interval(d, remainder_interval))
            # right remainder interval
            # right remainder equation

        # derivatives = new_equations

        all_derivatives += derivatives
    return all_derivatives


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


def replace_interval(loweredeq, interval):
    state = extract_le_state(loweredeq)

    old_ds = state['dspace']
    new_ds = DataSpace(replace_interval_group(old_ds.intervals, interval), old_ds.parts)

    old_is = state['ispace']
    new_is = IterationSpace(replace_interval_group(old_is.intervals, interval),
                                old_is.sub_iterators, old_is.directions)

    state['dspace'] = new_ds
    state['ispace'] = new_is

    return LoweredEq(loweredeq.lhs, loweredeq.rhs, **state)


def replace_interval_group(intervalgroup, interval):
    new_intervals = []

    for i in intervalgroup:
        if i.dim == interval.dim:
            new_intervals.append(interval)
        else:
            new_intervals.append(i)

    return IntervalGroup(new_intervals)
