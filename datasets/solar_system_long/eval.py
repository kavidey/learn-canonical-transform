# %%
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# from multiprocessing.dummy import Pool
from multiprocessing import Pool
from functools import partial
# %%
import rebound as rb
import reboundx
from celmech.nbody_simulation_utilities import get_canonical_heliocentric_orbits

# modified from celmech
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

# modified from celmech
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
        'd':np.zeros(shape),
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
        sim.integrator_synchronize() # need to syncronize whfast512 sim
        sim_results['time'][i] = sim.t
        orbits = get_orbits(sim)

        # calculate energy including GR potential
        rebx = reboundx.Extras(sim)
        gr = rebx.load_force("gr_potential")
        gr.params['c'] = 10065.32 # value of C from whfast512 integrator (https://github.com/hannorein/rebound/blob/6a4f95b58e71a69fdefd2ca5bd34097daf026655/src/integrator_whfast512.c#L895)
        sim_results['Energy'][i] = sim.energy() + rebx.gr_potential_potential(gr)

        for j,orbit in enumerate(orbits):
            sim_results['P'][j,i] = orbit.P
            sim_results['d'][j,i] = orbit.d
            sim_results['e'][j,i] = orbit.e
            sim_results['l'][j,i] = orbit.l
            sim_results['pomega'][j,i] = orbit.pomega
            sim_results['a'][j,i] = orbit.a
            sim_results['omega'][j,i] = orbit.omega
            sim_results['Omega'][j,i] = orbit.Omega
            sim_results['inc'][j,i] = orbit.inc
    return sim_results
# %%
if len(sys.argv) == 4:
    gen_npy_files = bool(sys.argv[1])
    d_thresh = float(sys.argv[1])
    e_thresh = float(sys.argv[2])
else:
    gen_npy_files = True
    d_thresh = 1
    e_thresh = 0.8
# %%
sim_dir = Path("sim")
npy_dir = Path("npy")

if gen_npy_files: npy_dir.mkdir(exist_ok=True)
# %%
def process_sim(d_thresh, e_thresh, f):
    npy_name = npy_dir/(f.stem+".npz")
    if not npy_name.exists():
        # return "", -1, -1, -1
        sim = get_simarchive_integration_results(str(f), coordinates='heliocentric')

        if gen_npy_files:
            np.savez_compressed(npy_name, **sim)
    else:
        sim = np.load(npy_name)

    unstable = np.bitwise_or(sim['d'][0] > d_thresh, sim['e'][0] > e_thresh)

    unstable_time = np.nonzero(unstable)[0]
    # print(unstable.sum())
    print(sim['d'][0].max(), sim['e'].max())
    if unstable_time.size > 0:
        # unstable_time = sim['time'][unstable_time[0]]
        unstable_time = unstable_time[0]
    else:
        unstable_time = -1
    
    return f.stem.split('_')[1], unstable_time, np.max(sim['d'][0]), np.max(sim['e'][0])

files = list(sim_dir.glob("*.bin"))

# change this number to make sure you are not hitting disk-io bottlenecks
with Pool(20) as p:
    results = list(tqdm(p.imap(partial(process_sim, d_thresh, e_thresh), files), total=len(files)))
# %%
df = pd.DataFrame(results, columns=['id', 'unstable_time', 'max_d', 'max_e'])

df['unstable_time'].max()
# %%
npy_f = npy_dir/(files[20].stem+".npz")
sim = np.load(npy_f)
# %%
import matplotlib.pyplot as plt

E0 = sim["Energy"][0]
Eerr = np.abs((E0 - sim["Energy"]) / E0)
plt.plot(sim['time']/(np.pi*2), Eerr)
plt.yscale("log")
plt.xscale("log")
# %%
for f in files:
    npy_f = npy_dir/(f.stem+".npz")
    if npy_f.exists():
        sim = np.load(npy_f)
        plt.plot(sim['time']/(np.pi*2), sim['e'][0], linewidth=0.1)
# %%
