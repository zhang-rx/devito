from sympy import Symbol

from devito import Eq, Operator, Function, TimeFunction, Inc, solve
from examples.seismic import PointSource, Receiver, ABC, Clayton


def laplacian(field, m, s, kernel):
    """
    Spacial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    m : Function or float
        Square slowness.
    s : float or Scalar
        The time dimension spacing.
    """
    if kernel not in ['OT2', 'OT4']:
        raise ValueError("Unrecognized kernel")

    biharmonic = field.biharmonic(1/m) if kernel == 'OT4' else 0
    return field.laplace + s**2/12 * biharmonic

def iso_stencil(field, kernel, model, s, **kwargs):
    """
    Stencil for the acoustic isotropic wave-equation:
    u.dt2 - H + damp*u.dt = 0.

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    m : Function or float
        Square slowness.
    s : float or Scalar
        The time dimension spacing.
    damp : Function
        The damping field for absorbing boundary condition.
    forward : bool
        The propagation direction. Defaults to True.
    q : TimeFunction, Function or float
        Full-space/time source of the wave-equation.
    """

    # Creat a temporary symbol for H to avoid expensive sympy solve
    H = Symbol('H')
    # Define time sep to be updated
    next = field.forward if kwargs.get('forward', True) else field.backward
    # Define PDE
    eq = model.m * field.dt2 - H - kwargs.get('q', 0)

    # Solve the symbolic equation for the field to be updated
    eq_time = solve(eq, next)

    # Get the spacial FD
    lap = laplacian(field, model.m, s, kernel)
    # return the Stencil with H replaced by its symbolic expression
    return [Eq(next, eq_time.subs({H: lap}), subdomain=model.grid.subdomains['phydomain'])]


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Construct a forward modelling operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    """

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    src = PointSource(name='src', grid=geometry.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, kernel, model, s)

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * s**2 / model.m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)

    BC = ABC(model, u)
    eq_abc = BC.abc

    # BC = Clayton(model, u, freesurface=False)
    # eq_abc = BC.abc_clayton

    # Substitute spacing terms to reduce flops
    return Operator(eqn + eq_abc + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4,
                    kernel='OT2', **kwargs):
    """
    Construct an adjoint modelling operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """

    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    srca = PointSource(name='srca', grid=model.grid, time_range=geometry.time_axis,
                       npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, kernel, model, s, forward=False)

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v)

    BC = ABC(model, v, forward=False)
    eq_abc = BC.abc

    # Substitute spacing terms to reduce flops
    return Operator(eqn + eq_abc + receivers + source_a, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True,
                     kernel='OT2', **kwargs):
    """
    Construct a gradient operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Option to store the entire (unrolled) wavefield.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """

    # Gradient symbol and wavefield symbols
    grad = Function(name='grad', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid, save=geometry.nt if save
                     else None, time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, kernel, model, s, forward=False)

    if kernel == 'OT2':
        gradient_update = Inc(grad, - u.dt2 * v)
    elif kernel == 'OT4':
        gradient_update = Inc(grad, - (u.dt2 + s**2 / 12.0 * u.biharmonic(m**(-2))) * v)
    # Add expression for receiver injection
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / model.m)
    BC = ABC(model, v, forward=False)
    eq_abc = BC.abc

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + eq_abc + [gradient_update], subs=model.spacing_map,
                    name='Gradient', **kwargs)


def BornOperator(model, geometry, space_order=4,
                 kernel='OT2', **kwargs):
    """
    Construct an Linearized Born operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """

    # Create source and receiver symbols
    src = Receiver(name='src', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Create wavefields and a dm field
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, kernel, model, s)
    eqn2 = iso_stencil(U, kernel, model, s, q=-dm*u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U)

    eq_abc = ABC(model, u).abc
    eq_abcl = ABC(model, U).abc

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + eq_abc + source + eqn2 + eq_abcl + receivers, subs=model.spacing_map,
                    name='Born', **kwargs)
