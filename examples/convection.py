from examples.cfd import init_smooth
import numpy as np
from devito import Grid, TimeFunction, INTERIOR, Eq, solve, Operator


# Some variable declarations
nx = 81
ny = 81
nt = 100
c = 1.
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
sigma = .2
dt = sigma * dx


grid = Grid(shape=(nx, ny), extent=(2., 2.))
u = TimeFunction(name='u', grid=grid)


eq = Eq(u.dt + c*u.dxl + c*u.dyl)

stencil = solve(eq, u.forward)

# Reset our data field and ICs in both buffers
init_smooth(field=u.data[0], dx=dx, dy=dy)
init_smooth(field=u.data[1], dx=dx, dy=dy)

# For defining BCs, we generally to explicitly set rows/columns
# in our field using an expression. We can use Devito's "indexed" 
# notation to do this:
x, y = grid.dimensions
t = grid.stepping_dim
#bc_left = Eq(u.indexed[t + 1, 0, y], 1.)
#bc_right = Eq(u.indexed[t + 1, nx-1, y], 1.)
#bc_top = Eq(u.indexed[t + 1, x, ny-1], 1.)
#bc_bottom = Eq(u.indexed[t + 1, x, 0], 1.)

# Now combine the BC expressions with the stencil to form operator.
expressions = [Eq(u.forward, stencil)]
#expressions += [bc_left, bc_right, bc_top, bc_bottom]
op = Operator(expressions=expressions, subs={x.spacing:1, y.spacing:1})
op(time=nt, dt=dt)


# Some small sanity checks for the testing framework
#assert (u.data[0, 45:55, 45:55] > 1.8).all()
