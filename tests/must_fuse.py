from devito import *

grid = Grid(shape=(4,))

a = Function(name='a', grid=grid)
b = Function(name='b', grid=grid)

x, = grid.dimensions

eqns = [Eq(a, b[x-1]),
        Eq(b, a)]

op = Operator(eqns)


# force it over different iteration spaces

x2 = Dimension(name='x')

eqns = [Eq(a, b[x-1]),
        Eq(b[x2], a[x2]),
        Eq(a, b[x-1])]

op = Operator(eqns)

