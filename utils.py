import numpy as np
import rebound as rb
from celmech.nbody_simulation_utilities import get_canonical_heliocentric_orbits
import jax.numpy as jnp
import jax

def integrate_leapfrog(func, y0, t, dt, args=()):
    """
    General symplectic leapfrog integrator (odeint-compatible API).

    Parameters
    ----------
    func : callable
        Function f(y, t, *args) returning dy/dt as 1D array of len(y0).
        First half of y are "positions" q, second half "momenta" p.
    y0 : array_like
        Initial state vector [q1, ..., qN, p1, ..., pN].
        Must have even length.
    t : array_like
        Times at which to output the solution.
    args : tuple, optional
        Additional parameters for func.
    dt : float, optional
        Internal integration timestep. Defaults to min(diff(t)) / 10.

    Returns
    -------
    y_out : ndarray
        Array of shape (len(t), len(y0)) containing y(t) at each output time.
    """

    t = np.asarray(t, dtype=float)
    t_end = t[-1]
    n_out = len(t)

    y0 = np.asarray(y0, dtype=float)
    ndim = len(y0)
    if ndim % 2 != 0:
        raise ValueError("y0 must have even length: [q1, ..., qN, p1, ..., pN]")

    n_dim = ndim // 2
    y_out = np.zeros((n_out, ndim))

    # Split into position and momentum arrays
    q = np.array(y0[:n_dim], dtype=float)
    p = np.array(y0[n_dim:], dtype=float)

    # Initial half-step momentum update
    dydt = np.asarray(func(np.concatenate([q, p]), 0.0, *args), dtype=float)
    dqdt, dpdt = dydt[:n_dim], dydt[n_dim:]
    p += 0.5 * dt * dpdt

    t_curr = 0.0
    next_index = 0

    # Integration loop
    for step in range(int(t_end / dt) + 2):
        # Output if reached/passed next requested time
        while next_index < n_out and t_curr >= t[next_index]:
            dydt = np.asarray(func(np.concatenate([q, p]), t_curr, *args), dtype=float)
            dqdt, dpdt = dydt[:n_dim], dydt[n_dim:]
            y_out[next_index, :n_dim] = q
            y_out[next_index, n_dim:] = p - 0.5 * dt * dpdt  # back to integer step
            next_index += 1
            if next_index == n_out:
                break
        if next_index == n_out:
            break

        # Full leapfrog update
        q += dt * p
        dydt = np.asarray(func(np.concatenate([q, p]), t_curr + 0.5 * dt, *args), dtype=float)
        dqdt, dpdt = dydt[:n_dim], dydt[n_dim:]
        p += dt * dpdt
        t_curr += dt

    return y_out

def get_eqns_motion(func):
    def wrapped_func(*args, **kwargs):
        grad_func = jax.grad(func, argnums=0)
        gradient = grad_func(*args, **kwargs)

        half_size = gradient.size // 2
        first_half, second_half = jnp.split(gradient, [half_size])

        modified_gradient = jnp.concatenate([second_half, -first_half])

        return modified_gradient
    return wrapped_func

def get_simarchive_integration_results(sa,coordinates='jacobi'):
    """
    Read a simulation archive and store orbital elements
    as arrays in a dictionary.

    Arguments
    ---------
    sa : rebound.Simulationarchive or str
     The simulation archive to read or the file name of the simulation
     archive file. Can also be a reboundx simulation archive.
    coordinates : str
        The coordinate system to use for calculating orbital elements. 
        This can be:
        | - 'jacobi' : Use Jacobi coordinates (including Jacobi masses)
        | - 'barycentric' : Use barycentric coordinates.
        | - 'heliocentric' : Use canonical heliocentric elements. 
        | The canonical cooridantes are heliocentric distance vectors.
        | The conjugate momenta are barycentric momenta.

    Returns
    -------
    sim_results : dict
        Dictionary containing time and orbital elements at each 
        snapshot of the simulation archive.
    """
    if type(sa) == str:
        sa = rb.Simulationarchive(sa)

    if type(sa) == rb.simulationarchive.Simulationarchive:
        return _get_rebound_simarchive_integration_results(sa,coordinates)
    raise TypeError("{} is not a rebound or reboundx simulation archive!".format(sa))

def _get_rebound_simarchive_integration_results(sa,coordinates):
    if coordinates == 'jacobi':
        get_orbits = lambda sim: sim.orbits(jacobi_masses=True)
    elif coordinates == 'heliocentric':
        get_orbits = get_canonical_heliocentric_orbits
    elif coordinates == 'barycentric':
        get_orbits = lambda sim: sim.orbits(sim.calculate_com())
    else: 
        raise ValueError("'Coordinates must be one of 'jacobi','heliocentric', or 'barycentric'")
    N = len(sa)
    sim0 = sa[0]
    Npl= sim0.N_real - 1
    shape = (Npl,N)
    sim_results = {
        'time':np.zeros(N),
        'P':np.zeros(shape),
        'e':np.zeros(shape),
        'l':np.zeros(shape),
        'inc':np.zeros(shape),
        'pomega':np.zeros(shape),
        'omega':np.zeros(shape),
        'Omega':np.zeros(shape),
        'a':np.zeros(shape),
        'Energy':np.zeros(N)
    }
    for i,sim in enumerate(sa):
        sim.synchronize()
        sim_results['time'][i] = sim.t
        orbits = get_orbits(sim)
        sim_results['Energy'][i] = sim.energy()
        for j,orbit in enumerate(orbits):
            sim_results['P'][j,i] = orbit.P
            sim_results['e'][j,i] = orbit.e
            sim_results['l'][j,i] = orbit.l
            sim_results['pomega'][j,i] = orbit.pomega
            sim_results['a'][j,i] = orbit.a
            sim_results['omega'][j,i] = orbit.omega
            sim_results['Omega'][j,i] = orbit.Omega
            sim_results['inc'][j,i] = orbit.inc
    return sim_results