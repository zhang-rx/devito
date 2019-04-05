from devito.symbolics.search import retrieve_indexed, search
from sympy import diff
from devito.types import TimeFunction, Dimension
from devito.ir.equations.equation import LoweredEq
from devito.ir.support.space import DataSpace, Interval, IntervalGroup, IterationSpace
from devito.equation import Inc


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
            new_eq = Inc(i_d, diff(e.rhs, i)*adjoint_output_fn)
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

        from IPython import embed
        embed()
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
    new_ispace = IterationSpace(shift_interval_group(ispace.intervals, mapper), ispace.sub_iterators, ispace.directions)

    state['dspace'] = new_ds
    state['ispace'] = new_ispace

    new_le = LoweredEq(le.lhs.subs(s_m), le.rhs.subs(s_m), **state)
    return new_le


def shift_interval_group(interval_group, mapper):
    new_intervals = []
    for interval in interval_group:
        if interval.dim in mapper:
            new_intervals.append(Interval(interval.dim, interval.lower+mapper[interval.dim],
                                          interval.upper+mapper[interval.dim]))
        else:
            new_intervals.append(interval)
    return IntervalGroup(new_intervals)


def q_dimension(expr):
    return isinstance(expr, Dimension)


def retrieve_dimension(expr, mode='unique', deep=False):
    """Shorthand to retrieve the Dimensions in ``expr``."""
    return search(expr, q_dimension, mode, 'dfs', deep)
