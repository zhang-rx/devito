from devito import *

grid = Grid(shape=(4, 4, 4))

f = TimeFunction(name='f', grid=grid, space_order=2)

eq = Eq(f.forward, f.laplace)

op = Operator(eq)
