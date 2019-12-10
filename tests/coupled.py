from devito import *

grid = Grid(shape=(4,))

# f = TimeFunction(name='f', grid=grid, space_order=2)
# g = TimeFunction(name='g', grid=grid)

f = Function(name='f', grid=grid, space_order=2)
g = Function(name='g', grid=grid)

x, = grid.dimensions

# f'(x) = f(x) + g(x)
# g(x) = f'(x) + g(x)
eqns = [Eq(f[x], f[x] + g[x-1] + 1),
        Eq(g[x], f[x-1] + 1)]


# flow dep -> must go together
# anti dep -> must be separated
# f'(x) = f(x) + g(x) + g'(x-1)
# g'(x) = f'(x) + g(x)
# eqns = [Eq(f, f + g + g[x+1]),
#        Eq(g[x], f + g)]

op = Operator(eqns)
