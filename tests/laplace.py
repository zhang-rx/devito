from devito import *

grid = Grid(shape=(4, 4))

# f = TimeFunction(name='f', grid=grid, space_order=2)
# g = TimeFunction(name='g', grid=grid)

f = Function(name='f', grid=grid, space_order=2)
g = Function(name='g', grid=grid)

# f(t+1, x, y, z) := f.dx^2 + f.dy^2 + f.dz^2
# eq = Eq(f.forward, f.laplace)

eqns = [Eq(f, f + 1),
        Eq(g, f.dxr + 1)]

op = Operator(eqns)

op.apply(...)


# grid = Grid(shape=(...), dimensions=(.....))
# f2 = TimeFunction(name='f2', grid=grid, dimensions=(t, x, y), shape=())
# op.apply(f=f2)
