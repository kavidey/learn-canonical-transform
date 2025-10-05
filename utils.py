import numpy as np
import rebound as rb
from celmech.nbody_simulation_utilities import get_canonical_heliocentric_orbits

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
        sim.integrator_synchronize()
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