from __future__ import division

import numpy as np

from devito import Dimension, Function, SubDimension, Eq, first_derivative
from devito.exceptions import InvalidArgument
from devito.finite_difference import transpose

from sympy import sqrt, solve, finite_diff_weights

__all__ = ['ABC', 'Clayton']


# assuming (t/time, x, y, z), for now, generalization gonna be tricky

class ABC(object):
    """
    Absrobing boundary layer for second-order scalar acoustic wave equations
    :param model : model structure containing the boundary layer size and velocity model
    :param field : propagated field as a TimdeData object
    :param m : square slowness as a DenseDat object
    :param taxis : Forward or Backward, defines the propagation axis
    """

    def __init__(self, model, field, forward=True, **kwargs):
        self.nbpml = int(model.nbpml)
        self.full_shape = model.shape_domain
        self.ndim = model.dim
        self.indices = field.grid.dimensions
        self.field = field
        self.tindex = self.field.grid.time_dim
        self.m = model.m
        self.forward = forward
        self.freesurface = kwargs.get("freesurface", False)

    def abc_eq(self, abc_dim, left=True):
        """
        Equation of the absorbing boundary condition as a complement of the PDE
        :param val: symbolic value of the dampening profile
        :return: Symbolic equation inside the boundary layer
        """
        # Define time sep to be updated
        next = self.field.forward if self.forward else self.field.backward
        # Define PDE
        eta = self.damp_profile_init(abc_dim, left=left)
        eq = self.m * self.field.dt2 - self.field.laplace
        eq += eta * self.field.dt if self.forward else -eta * self.field.dt
        # Solve the symbolic equation for the field to be updated
        eq_time = solve(eq, next, rational=False, simplify=False)[0]
        # return the Stencil with H replaced by its symbolic expression
        return Eq(next, eq_time).subs({abc_dim.parent: abc_dim})


    @property
    def free_surface(self):
        """
        Free surface expression. Mirrors the negative wavefield above the sea level
        :return: Symbolic equation of the free surface
        """
        dim = self.field.indices[-1]
        dim_abc_left = SubDimension.left(name='abc_'+ dim.name + '_left', parent=dim, thickness=self.nbpml)
        dim_abc_right = SubDimension.right(name='abc_'+ dim.name + '_right', parent=dim, thickness=self.nbpml)
        next = self.field.forward if self.forward else self.field.backward
        return [Eq(next.subs({dim: dim_abc_left}), - next.subs({dim: 2*self.nbpml - dim_abc_left})),
                self.abc_eq(dim_abc_right, left=False)]

    @property
    def abc(self):
        """
        Complete set of expressions for the ABC layers
        :return:
        """
        if self.ndim == 1:
            return self.damp_x
        elif self.ndim == 2:
            return self.damp_2d
        elif self.ndim == 3:
            return self.damp_3d
        else:
            raise InvalidArgument("Unsupported model shape")

    def damp_profile_init(self, abc_dim, left=True):
        """
        Dampening profile along a single direction
        :return:
        """
        positions = np.linspace(1.0, 0.0, self.nbpml) if left else np.linspace(0.0, 1.0, self.nbpml)
        coeff = 1.5 * np.log(1.0 / 0.001) / (40.)
        profile = [coeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi)) for pos in positions]
        damp = Function(name="damp" + abc_dim.name, shape=(self.nbpml,), dimensions=(abc_dim,), dtype=np.float32, space_order=0)
        damp.data[:] = damp.dtype(profile / self.field.grid.spacing[0])
        return damp.subs({damp.indices[0]: damp.indices[0]- damp.indices[0].symbolic_start})

    @property
    def damp_x(self):
        """
        Dampening profile along x
        :return:
        """
        dim = self.indices[0]
        dim_abc_left = SubDimension.left(name='abc_'+ dim.name + '_left', parent=dim, thickness=self.nbpml)
        dim_abc_right = SubDimension.right(name='abc_'+ dim.name + '_right', parent=dim, thickness=self.nbpml)
        return [self.abc_eq(dim_abc_left), self.abc_eq(dim_abc_right, left=False)]

    @property
    def damp_y(self):
        """
        Dampening profile along y
        :return:
        """
        dim = self.indices[1]
        dim_abc_left = SubDimension.left(name='abc_'+ dim.name + '_left', parent=dim, thickness=self.nbpml)
        dim_abc_right = SubDimension.right(name='abc_'+ dim.name + '_right', parent=dim, thickness=self.nbpml)
        return [self.abc_eq(dim_abc_left), self.abc_eq(dim_abc_right, left=False)]

    @property
    def damp_z(self):
        """
        Dampening profile along y
        :return:
        """
        dim = self.indices[2]
        dim_abc_left = SubDimension.left(name='abc_'+ dim.name + '_left', parent=dim, thickness=self.nbpml)
        dim_abc_right = SubDimension.right(name='abc_'+ dim.name + '_right', parent=dim, thickness=self.nbpml)
        return [self.abc_eq(dim_abc_left), self.abc_eq(dim_abc_right, left=False)]

    @property
    def damp_2d(self):
        """
        Dampening profiles in 2D w/ w/o freesurface
        :return:
        """
        return self.damp_x + (self.free_surface if self.freesurface else self.damp_y)

    @property
    def damp_3d(self):
        """
        Dampening profiles in 2D w/ w/o freesurface
        :return:
        """
        return self.damp_x + self.damp_y +\
            (self.free_surface if self.freesurface else self.damp_z)


class Clayton(object):
        """
        file:///nethome/mlouboutin3/Downloads/174031.pdf
        Robin boundary layer for second-order scalar acoustic wave equations
        :param model : model structure containing the boundary layer size and velocity model
        :param field : propagated field as a TimdeData object
        :param m : square slowness as a DenseDat object
        :param taxis : Forward or Backward, defines the propagation axis
        """

        def __init__(self, model, field, forward=True, **kwargs):
            self.nbpml = int(model.nbpml)
            self.full_shape = model.shape_domain
            self.ndim = model.dim
            self.indices = field.grid.dimensions
            self.field = field
            self.tindex = self.field.grid.time_dim
            self.m = model.m
            self.forward = forward
            self.freesurface = kwargs.get("freesurface", False)

        @property
        def abc_clayton(self):
            bcs = []
            for dim in reversed(self.indices[:-1]):
                bcs += self.Clayton_single(dim, "left")
                bcs += self.Clayton_single(dim, "right")
            if self.freesurface:
                next = self.field.forward if self.forward else self.field.backward
                subdim = SubDimension.left(name=self.indices[-1].name+"_fs", parent=self.indices[-1], thickness=self.nbpml)
                bcs += [Eq(next.subs({self.indices[-1]: subdim}), - next.subs({self.indices[-1]: 2*self.nbpml - subdim}))]
                bcs += self.Clayton_single(self.indices[-1], "right")
            else:
                bcs += self.Clayton_single(self.indices[-1], "left")
                bcs += self.Clayton_single(self.indices[-1], "right")
            return bcs

        def Clayton_single(self, dim, side):
            if side == "left":
                sign = 1
                subdim = SubDimension.left(name=dim.name+"_"+ side, parent=dim, thickness=self.nbpml)
                # pxt - 1/v*ptt + v/2 * pyy
            elif side == "right":
                sign = -1
                subdim = SubDimension.right(name=dim.name+"_" + side, parent=dim, thickness=self.nbpml)
                # pxt + 1/v*ptt - v/2*pyy
            else:
                raise Exception("pick a side, 'left' or 'right'")
            partial_lap = self.field.laplace - getattr(self.field, 'd%s2' % dim)

            ind = [(dim + i * dim.spacing * sign) for i in range(0, int(self.field.space_order/ 2) +1)]
            c = finite_diff_weights(1, ind, dim)
            c = c[-1][-1]
            dudx = sum([c[i]*self.field.subs({dim: ind[i]}) for i in range(len(ind))])
            cross_derivative = first_derivative(dudx, dim=self.field.indices[0], order=1)

            v = sqrt(1/self.m)
            damp = self.damp_profile_init(subdim, left=side)
            pde = 1 / v * self.field.dt2 -  .5 * v * partial_lap - sign * cross_derivative



            step_single = Eq(self.field.forward, solve(pde, self.field.forward, rational=False)[0])

            step_sub = step_single.subs(dim, subdim)
            return [step_sub]

        def damp_profile_init(self, abc_dim, left="left"):
            """
            Dampening profile along a single direction
            :return:
            """
            positions = np.linspace(.7, 0.2, self.nbpml) if left=="left" else np.linspace(0.2, .7, self.nbpml)
            damp = Function(name="damp" + abc_dim.name, shape=(self.nbpml,), dimensions=(abc_dim,), dtype=np.float32, space_order=0)
            damp.data[:] = positions**2
            return damp.subs({damp.indices[0]: damp.indices[0]- damp.indices[0].symbolic_start})
