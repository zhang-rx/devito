from __future__ import division

import numpy as np

from devito import Dimension, Function, SubDimension, Eq
from devito.exceptions import InvalidArgument

from sympy import sqrt, solve

__all__ = ['ABC']


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
        self.p_abc = Dimension(name="abc")
        self.ndim = model.dim
        self.indices = field.grid.dimensions
        self.field = field
        self.tindex = self.field.grid.time_dim
        self.m = model.m
        self.forward = forward
        self.freesurface = kwargs.get("freesurface", False)
        self.damp_profile = self.damp_profile_init()

    @property
    def abc_eq(self):
        """
        Equation of the absorbing boundary condition as a complement of the PDE
        :param val: symbolic value of the dampening profile
        :return: Symbolic equation inside the boundary layer
        """
        # Define time sep to be updated
        next = self.field.forward if self.forward else self.field.backward
        # Define PDE
        eq = self.m * self.field.dt2 - self.field.laplace + self.damp_profile * self.field.dt
        # Solve the symbolic equation for the field to be updated
        eq_time = solve(eq, next, rational=False, simplify=False)[0]
        # return the Stencil with H replaced by its symbolic expression
        return Eq(next, eq_time)


    @property
    def free_surface(self):
        """
        Free surface expression. Mirrors the negative wavefield above the sea level
        :return: SYmbolic equation of the free surface
        """
        ind_fs = self.field.indices[-1]
        next = self.field.forward if self.forward else self.field.backward
        return [Eq(next.subs({ind_fs: self.p_abc}),
                   - next.subs({ind_fs: 2*self.nbpml - self.p_abc}))]

    @property
    def abc(self):
        """
        Complete set of expressions for the ABC layers
        :return:
        """
        if self.ndim == 2:
            return self.damp_2d()
        elif self.ndim == 3:
            return self.damp_3d()
        else:
            raise InvalidArgument("Unsupported model shape")

    def damp_profile_init(self):
        """
        Dampening profile along a single direction
        :return:
        """
        coeff = 1.5 * np.log(1.0 / 0.001) / (40.)
        profile = [coeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi)) for pos in np.linspace(1.0, 0.0, self.nbpml)]
        # second order would be 1/250*pos)**2 *(1 + 1/2*(1/250*pos)*(1/250*pos))
        # profile = [(1 / 250 * pos) * (1 / 250 * pos) for pos in range(self.nbpml-1,0,-1)]
        damp = Function(name="damp", shape=(self.nbpml,), dimensions=(self.p_abc,), dtype=np.float32,
                        space_order=0)
        damp.data[:] = profile / self.field.grid.spacing[0]
        return damp

    def damp_x(self):
        """
        Dampening profile along x
        :return:
        """
        return [self.abc_eq.subs({self.indices[0]: self.p_abc}),
                self.abc_eq.subs({self.indices[0]: self.full_shape[0] - 1 - self.p_abc})]

    def damp_y(self):
        """
        Dampening profile along y
        :return:
        """
        return [self.abc_eq.subs({self.indices[1]: self.p_abc}),
                self.abc_eq.subs({self.indices[1]: self.full_shape[1] - 1 - self.p_abc})]

    def damp_z(self):
        """
        Dampening profile along y
        :return:
        """
        return [self.abc_eq.subs({self.indices[2]: self.p_abc}),
                self.abc_eq.subs({self.indices[2]: self.full_shape[2] - 1 - self.p_abc})]

    def damp_2d(self):
        """
        Dampening profiles in 2D w/ w/o freesurface
        :return:
        """
        return self.damp_x() + (self.free_surface if self.freesurface else self.damp_y())

    def damp_3d(self):
        """
        Dampening profiles in 2D w/ w/o freesurface
        :return:
        """
        return self.damp_x() + self.damp_y() +\
            (self.free_surface if self.freesurface else self.damp_z())
