from devito import Eq, Grid, TimeFunction, Operator


grid = Grid(shape=(4, 4))
t = grid.stepping_dim
x, y = grid.dimensions
u = TimeFunction(name='u', grid=grid)
u.data[:]
u_i = u.indexed
eq = Eq(u_i[t+1, x, y], 4*u_i[t, x, y] + 2*u_i[t, x-1, y] + 3*u_i[t, x+1, y])
# eq2 = Eq(u.forward, u + 1)
op = Operator(eq, adjoint=True)
print("In the main program now")
print(op)
